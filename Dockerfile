# ---- Backend + Static Frontend ----
FROM node:20-alpine AS app
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --production || yarn install --production --ignore-engines || true
COPY server.js crttrade.html ./
EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]
