from slowapi import Limiter
from slowapi.util import get_remote_address

# Inicializa o limitador usando o IP do cliente como chave de identificação
# Se estivesse atrás de um proxy reverso (Nginx/Cloudflare) em produção real,
# talvez precisasse ajustar o get_remote_address, mas para seu setup serve.
limiter = Limiter(key_func=get_remote_address)