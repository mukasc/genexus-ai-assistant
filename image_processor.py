import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_path

# Certifique-se de que load_dotenv("keys.env") está correto
load_dotenv("keys.env") 
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY não encontrada. Verifique keys.env.")

# Inicializa o cliente Gemini
gemini_client = genai.Client(api_key=API_KEY)

def describe_image_with_gemini(image_path_or_bytes):
    """Envia uma imagem para o Gemini Vision para gerar uma descrição detalhada."""
    
    # Prompt de engenharia para obter uma descrição técnica e útil para RAG
    prompt = (
        "Descreva esta imagem de forma técnica e concisa para um desenvolvedor GeneXus. "
        "Foque em elementos como nomes de objetos, atributos, diagramas de fluxo de dados, "
        "propriedades ou código visível. Comece a descrição com '[IMAGEM DESCRITA]: '."
    )
    
    # Se for um caminho, abre a imagem; caso contrário, assume que são bytes
    if isinstance(image_path_or_bytes, str):
        img = Image.open(image_path_or_bytes)
    else:
        img = Image.open(image_path_or_bytes)
        
    try:
        # Usamos o modelo Pro Vision para descrição
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', # Gemini-2.5-Flash é multimodal e mais rápido
            contents=[prompt, img]
        )
        return response.text
    except Exception as e:
        print(f"Erro ao descrever imagem com Gemini: {e}")
        return "[IMAGEM NÃO DESCRITA DEVIDO A ERRO]"


def extract_and_describe_from_pdf(pdf_path, output_dir="./processed_text"):
    """Simula a extração de texto e a descrição das imagens para enriquecimento."""
    
    print(f"Processando {pdf_path}...")
    
    # Esta é uma simplificação. A extração real em um PDF complexo é difícil.
    # Aqui, convertemos cada página em uma imagem e descrevemos essa imagem.
    
    pages = convert_from_path(pdf_path)
    enriched_text = ""
    
    for i, page_image in enumerate(pages):
        # 1. Obter o texto da página (simulação/melhoria necessária para texto real)
        # O método ideal aqui seria usar uma biblioteca OCR avançada para obter o layout
        # Mas para o protótipo, focamos apenas na descrição visual.
        
        # 2. Descrever a Imagem da Página inteira (lenta, mas eficaz)
        with BytesIO() as output:
            page_image.save(output, format="PNG")
            image_bytes = output.getvalue()
        
        # Envia a página (como imagem) para o Gemini
        description = describe_image_with_gemini(BytesIO(image_bytes))
        
        # 3. Adiciona a descrição ao texto enriquecido
        enriched_text += f"\n\n--- INÍCIO DO CONTEÚDO VISUAL PÁGINA {i+1} ---\n{description}\n--- FIM DO CONTEÚDO VISUAL ---\n\n"
        
        # Em um cenário real, você adicionaria o texto extraído por OCR aqui
        # enriched_text += text_extractor.extract(pdf_path, page=i)

    # Salva o texto enriquecido (com descrições) em um arquivo .txt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_filename = os.path.join(output_dir, os.path.basename(pdf_path).replace(".pdf", "_enriched.txt"))
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(enriched_text)
        
    print(f"Conteúdo enriquecido salvo em: {output_filename}")
    return output_filename

# Exemplo de execução (para teste)
if __name__ == "__main__":
    # Assumindo que você tem um PDF de teste na pasta docs
    pdf_to_process = os.path.join("docs", "seu_manual_genexus.pdf")
    
    # Você precisará trocar 'seu_manual_genexus.pdf' por um arquivo real
    if os.path.exists(pdf_to_process):
        extract_and_describe_from_pdf(pdf_to_process)
    else:
        print("Caminho do PDF de teste não encontrado. Crie um PDF ou ajuste o caminho.")