FROM node:20-alpine AS builder

WORKDIR /app

COPY nba-recs-frontend/package*.json ./
RUN npm ci

COPY nba-recs-frontend/ .
RUN npm run build

FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
