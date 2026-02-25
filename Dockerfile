# ---- Backend + Static Frontend ----
FROM node:20-alpine AS app
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --omit=dev
COPY server.js crttrade.html ./
# Redeploy stamp: 2026-02-23T12:00:00Z - bump to force redeploy
EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]
