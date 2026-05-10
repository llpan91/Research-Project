# Stage 1: Build Hugo site
FROM hugomods/hugo:exts-0.140.0 AS builder

WORKDIR /src
COPY . .

RUN hugo --minify

# Stage 2: Serve with nginx
FROM nginx:alpine

COPY --from=builder /src/public /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
