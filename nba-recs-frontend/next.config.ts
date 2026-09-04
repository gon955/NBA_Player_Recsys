import type { NextConfig } from "next";

// Deploy target is a static export: every page in this app is a client
// component and all data is fetched from the backend in the browser, so there
// is no server to run. `next build` emits `out/`, which is what Cloudflare
// Pages serves.
//
// docker-compose still wants a Node server for local dev, so its Dockerfile
// sets NEXT_OUTPUT=standalone to get the previous behaviour back.
const nextConfig: NextConfig = {
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : "export",
};

export default nextConfig;
