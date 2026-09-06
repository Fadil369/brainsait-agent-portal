/**
 * BrainSAIT Agent Portal — unified voice clinical AI.
 *
 * Combines:
 * 1. Voice input (ElevenLabs STT → text)
 * 2. Clinical RAG search (AI Search → grounded answers)
 * 3. OID verification (provider lookup via OID graph)
 * 4. Claim rejection analysis (D1 query)
 * 5. Voice output (ElevenLabs TTS → spoken answer)
 * 6. Analytics (every query tracked per agent/encounter)
 *
 * Architecture:
 *   Voice → STT → Intent Detection → Parallel RAG/OID/Claims → Synthesize → TTS → Voice
 */

export interface Env {
  AI: Ai;
  AI_SEARCH: AiSearchNamespace;
  HIS_DB: D1Database;
  WAE: AnalyticsEngineDataset;
  SESSIONS_KV: KVNamespace;
  ELEVENLABS_API_KEY?: string;
  ELEVENLABS_VOICE_ID?: string;
}

const DEFAULT_VOICE = "EXAVITQu4vr4xnSDxMaL"; // Sarah
const TTS_MODEL = "eleven_turbo_v2_5";

// ---- Intent Detection ----

type Intent = "search" | "verify_provider" | "check_claim" | "greeting" | "unknown";

function detectIntent(text: string): Intent {
  const t = text.toLowerCase();
  if (/^(hi|hello|hey|مرحبا|السلام|أهلا)/.test(t)) return "greeting";
  if (/verify|check provider|lookup provider|search provider|spid|sa-phy|provider status|doctor verified|physician verified|trust level|تحقق.*طبيب|بحث.*مقدم/.test(t)) return "verify_provider";
  if (/check claim|claim status|why.*reject|rejection reason|denied.*claim|prior.auth.*required|مطالبة.*حالة|سبب.*رفض/.test(t)) return "check_claim";
  return "search";
}

// ---- OID Verification ----

async function verifyProvider(env: Env, query: string): Promise<string> {
  // Try to extract SPID from query
  const spidMatch = query.match(/SA-[A-Z]{3}-\d{6}/i) || query.match(/SA-[A-Z]{3}-\d{3,6}/i);
  const spid = spidMatch?.[0]?.toUpperCase();

  if (!spid) {
    return `No SPID found in query. Please provide a provider SPID like SA-PHY-000001 to verify.`;
  }

  try {
    const res = await fetch(`https://oid-line.brainsait-fadil.workers.dev/v1/verify/${spid}`);
    const data = await res.json() as any;
    if (data.verified) {
      return `Provider ${spid} is VERIFIED. Status: ${data.status}, Trust level: ${data.trust || "BIAL"}, OID: ${data.oid || "N/A"}. Verification page: https://verify.brainsait.org/${spid}`;
    }
    return `Provider ${spid} could not be verified. Status: ${data.status || "unknown"}.`;
  } catch (e: any) {
    return `Verification service temporarily unavailable: ${e.message}`;
  }
}

// ---- Claim Rejection Analysis ----

async function checkClaim(env: Env, query: string): Promise<string> {
  try {
    // Query rejection codes from D1
    const result = await env.HIS_DB.prepare(
      `SELECT code, COUNT(*) as count FROM rejection_codes GROUP BY code ORDER BY count DESC LIMIT 10`
    ).all<{ code: string; count: number }>();

    if (!result.results || result.results.length === 0) {
      return "No rejection data found in the clinical database. The rejection learner may need more claim submissions to mine patterns.";
    }

    const topCodes = result.results.slice(0, 5);
    let response = "Top rejection codes from recent claims:\n";
    for (const r of topCodes) {
      response += `- ${r.code}: ${r.count} occurrences\n`;
    }
    response += "\nThese are the most common rejection patterns. To check a specific claim, provide the claim ID or rejection code.";

    // Check for specific code in query
    const codeMatch = query.match(/\b[A-Z]{2,3}-\d+-\d+\b/);
    if (codeMatch) {
      const code = codeMatch[0];
      const details = await env.HIS_DB.prepare(
        `SELECT * FROM rejection_codes WHERE code = ?`
      ).bind(code).first();
      if (details) {
        response = `Rejection code ${code}: ${JSON.stringify(details)}`;
      } else {
        response = `Rejection code ${code} not found in database. Available codes: ${topCodes.map(r => r.code).join(", ")}`;
      }
    }

    return response;
  } catch (e: any) {
    return `Claim analysis temporarily unavailable: ${e.message}`;
  }
}

// ---- Clinical RAG Search ----

async function clinicalSearch(env: Env, query: string): Promise<string> {
  try {
    const handle = env.AI_SEARCH.get("brainsait-ai-search");
    const result = await handle.search({
      query,
      ai_search_options: { retrieval: { max_num_results: 3 } },
    });
    const chunks = (result.chunks || []).map((c: any) => c.text).join("\n\n");
    if (!chunks) return "No clinical information found for this query.";
    return chunks;
  } catch (e: any) {
    return `Clinical search temporarily unavailable: ${e.message}`;
  }
}

// ---- ElevenLabs TTS ----

async function tts(env: Env, text: string): Promise<Response> {
  if (!env.ELEVENLABS_API_KEY) {
    return Response.json({ ok: false, error: "TTS not configured" }, { status: 503 });
  }
  const voice = env.ELEVENLABS_VOICE_ID || DEFAULT_VOICE;
  const res = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voice}?output_format=mp3_44100_128`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "xi-api-key": env.ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({ text, model_id: TTS_MODEL }),
    }
  );
  if (!res.ok) {
    return Response.json({ ok: false, error: `TTS error: ${res.status}` }, { status: 502 });
  }
  return new Response(res.body, {
    headers: {
      "Content-Type": "audio/mpeg",
      "Content-Disposition": 'attachment; filename="response.mp3"',
    },
  });
}

// ---- Analytics ----

function track(env: Env, data: { intent: string; query: string; responseLength: number; latencyMs: number; spid?: string }) {
  env.WAE.writeDataPoint({
    blobs: [data.intent, data.query.slice(0, 200), data.spid || ""],
    doubles: [data.responseLength, data.latencyMs, Date.now()],
  });
}

// ---- Main Chat Handler ----

async function handleChat(env: Env, query: string, speak: boolean): Promise<Response> {
  const started = Date.now();
  const intent = detectIntent(query);
  let response = "";

  switch (intent) {
    case "greeting":
      response = "Welcome to BrainSAIT Agent Portal. I can help you verify healthcare providers, check claim status, or search the clinical knowledge base. How can I assist you today? مرحباً بك في بوابة وكيل Brainsait. يمكنني مساعدتك في التحقق من مزودي الرعاية الصحية، أو التحقق من حالة المطالبة، أو البحث في قاعدة المعرفة السريرية.";
      break;
    case "verify_provider":
      response = await verifyProvider(env, query);
      break;
    case "check_claim":
      response = await checkClaim(env, query);
      break;
    case "search":
    default:
      response = await clinicalSearch(env, query);
      break;
  }

  const latencyMs = Date.now() - started;
  track(env, { intent, query, responseLength: response.length, latencyMs });

  if (speak) {
    const audioRes = await tts(env, response);
    if (audioRes.ok) {
      return new Response(JSON.stringify({
        ok: true,
        intent,
        query,
        response: response.slice(0, 500),
        latencyMs,
        audioUrl: "/tts?text=" + encodeURIComponent(response.slice(0, 500)),
      }), {
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  return Response.json({ ok: true, intent, query, response: response.slice(0, 500), latencyMs });
}

// ---- Exported Handler ----

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
      });
    }

    // Health
    if (path === "/" || path === "/health") {
      return Response.json({
        ok: true,
        service: "brainsait-agent-portal",
        version: "1.0.0",
        status: "unified voice clinical AI — live",
        capabilities: [
          "clinical search (AI Search)",
          "provider verification (OID)",
          "claim rejection analysis (D1)",
          "voice TTS (ElevenLabs)",
          "analytics tracking",
        ],
        endpoints: ["/health", "/chat", "/tts", "/verify", "/claims", "/search"],
      });
    }

    // Chat — main orchestration endpoint
    if (path === "/chat" && request.method === "POST") {
      const body = await request.json().catch(() => ({})) as any;
      const query = body.query || "";
      const speak = body.speak !== false;
      if (!query) return Response.json({ ok: false, error: "query required" }, { status: 400 });
      return handleChat(env, query, speak);
    }

    // TTS — direct text-to-speech
    if (path === "/tts" && request.method === "POST") {
      const body = await request.json().catch(() => ({})) as any;
      const text = body.text || "";
      if (!text) return Response.json({ ok: false, error: "text required" }, { status: 400 });
      return tts(env, text);
    }

    // Verify — direct provider verification
    if (path === "/verify") {
      const spid = url.searchParams.get("spid") || "";
      if (!spid) return Response.json({ ok: false, error: "spid required" }, { status: 400 });
      const result = await verifyProvider(env, `verify ${spid}`);
      return Response.json({ ok: true, spid, result });
    }

    // Claims — direct claim analysis
    if (path === "/claims") {
      const query = url.searchParams.get("q") || "";
      if (!query) return Response.json({ ok: false, error: "q required" }, { status: 400 });
      const result = await checkClaim(env, query);
      return Response.json({ ok: true, query, result });
    }

    // Search — direct clinical search
    if (path === "/search") {
      const query = url.searchParams.get("q") || "";
      if (!query) return Response.json({ ok: false, error: "q required" }, { status: 400 });
      const result = await clinicalSearch(env, query);
      return Response.json({ ok: true, query, result });
    }

    return Response.json({
      ok: false,
      error: "Not found — try /health, /chat, /tts, /verify, /claims, /search",
    }, { status: 404 });
  },
};
