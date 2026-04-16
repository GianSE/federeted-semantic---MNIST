/**
 * classifier.routes.js
 * --------------------
 * Proxy routes for classifier training and evaluation.
 */

export async function registerClassifierRoutes(app, mlServiceUrl) {
  app.post("/api/classifier/train", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/classifier/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request.body || {}),
    });
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.get("/api/classifier/status", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/classifier/status`);
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.post("/api/classifier/stop", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/classifier/stop`, { method: "POST" });
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.post("/api/classifier/logs/clear", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/classifier/logs/clear`, { method: "POST" });
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.get("/api/classifier/logs/stream", async (request, reply) => {
    const upstream = await fetch(`${mlServiceUrl}/classifier/logs/stream`);

    reply.raw.writeHead(upstream.status, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    if (!upstream.body) {
      reply.raw.end();
      return reply;
    }

    const reader = upstream.body.getReader();
    const pump = async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        reply.raw.write(Buffer.from(value));
      }
      reply.raw.end();
    };

    pump().catch(() => reply.raw.end());
    return reply;
  });

  app.get("/api/classifier/results/latest", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/classifier/results/latest`);
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.get("/api/classifier/results/experiments", async (request, reply) => {
    const response = await fetch(`${mlServiceUrl}/classifier/results/experiments`);
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.get("/api/classifier/results/experiments/:experimentId", async (request, reply) => {
    const { experimentId } = request.params;
    const response = await fetch(`${mlServiceUrl}/classifier/results/experiments/${encodeURIComponent(experimentId)}`);
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });

  app.get("/api/classifier/results/artifact/:experimentId/*", async (request, reply) => {
    const { experimentId } = request.params;
    const wildcardPath = request.params["*"];
    const response = await fetch(
      `${mlServiceUrl}/classifier/results/artifact/${encodeURIComponent(experimentId)}/${wildcardPath}`
    );

    if (!response.body) {
      const text = await response.text();
      reply.code(response.status).send(text);
      return;
    }

    reply.raw.writeHead(response.status, {
      "Content-Type": response.headers.get("content-type") || "application/octet-stream",
      "Cache-Control": "no-cache",
    });

    const reader = response.body.getReader();
    const pump = async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        reply.raw.write(Buffer.from(value));
      }
      reply.raw.end();
    };

    pump().catch(() => reply.raw.end());
    return reply;
  });
}
