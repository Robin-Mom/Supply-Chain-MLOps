FROM nginx:latest

COPY nginx.conf /etc/nginx/nginx.conf
COPY certs/ /etc/nginx/certs/
COPY .htpasswd /etc/nginx/.htpasswd
COPY auth.conf /etc/nginx/auth.conf
COPY proxy_headers.conf /etc/nginx/proxy_headers.conf
EXPOSE 80
EXPOSE 443
EXPOSE 8081
CMD ["nginx", "-g", "daemon off;"]