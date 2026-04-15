export async function registerWeightsRoutes(app, mlServiceUrl) {
  app.get("/api/weights", async (request, reply) => {
    const { dataset, model } = request.query || {};
    const params = new URLSearchParams();
    if (dataset) params.set("dataset", dataset);
    if (model) params.set("model", model);

    const response = await fetch(`${mlServiceUrl}/weights/list?${params.toString()}`);
    const text = await response.text();
    reply.code(response.status).type(response.headers.get("content-type") || "application/json").send(text);
  });
}
