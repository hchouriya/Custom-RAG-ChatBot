# syntax=docker/dockerfile:1
# Simple Node 22 image: install → build → start.
# When the app enables `output: 'standalone'` in next.config, this can be
# swapped for a leaner multi-stage copy of `.next/standalone`.
FROM node:22-alpine

WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .
RUN npm run build

RUN addgroup --system --gid 10001 nodejs \
    && adduser --system --uid 10001 nextjs \
    && chown -R nextjs:nodejs /app
USER nextjs

EXPOSE 3000
CMD ["npm", "run", "start"]
