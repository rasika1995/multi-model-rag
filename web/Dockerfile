# UI: Next.js production build served by `next start`.
FROM node:20-alpine AS build
WORKDIR /web
# Optional corporate/proxy root CA: drop *.crt files in web/docker-certs/ (see README).
COPY docker-certs/ /certs/
RUN cat /certs/*.crt > /certs/extra.pem 2>/dev/null; true
ENV NODE_EXTRA_CA_CERTS=/certs/extra.pem NEXT_TELEMETRY_DISABLED=1
COPY package.json ./
RUN npm install --no-audit --no-fund
COPY . .
# The browser calls the API directly, so this is the API address as seen from the user's browser.
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
RUN npm run build

FROM node:20-alpine
WORKDIR /web
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
COPY --from=build /web/package.json ./
COPY --from=build /web/node_modules ./node_modules
COPY --from=build /web/.next ./.next
EXPOSE 3000
CMD ["npx", "next", "start", "-p", "3000"]
