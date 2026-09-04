FROM node:20-alpine AS builder

WORKDIR /app

COPY nba-recs-frontend/package*.json ./
RUN npm ci

COPY nba-recs-frontend/ .
ARG NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
# next.config.ts defaults to a static export for Cloudflare Pages; this image
# serves the app from a Node server instead, so ask for the standalone build.
ENV NEXT_OUTPUT=standalone
RUN npm run build

FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["sh", "-c", "HOST=0.0.0.0 node server.js -p ${PORT:-3000}"]
