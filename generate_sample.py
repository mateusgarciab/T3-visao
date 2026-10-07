import cv2
import numpy as np

def generate_sample_document(output_path="entrada/documento.jpg"):
    # 1. Create a clean A4-like flat document image (800x1131)
    doc_w, doc_h = 800, 1131
    doc = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 255

    # Draw border lines on document
    cv2.rectangle(doc, (30, 30), (doc_w - 30, doc_h - 30), (40, 40, 40), 3)
    cv2.rectangle(doc, (40, 40), (doc_w - 40, doc_h - 40), (200, 200, 200), 1)

    # Header section
    cv2.putText(doc, "RELATORIO TECNICO DE COMPUTACAO GRAFICA", (70, 100),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 100), 2)
    cv2.putText(doc, "UNIVERSIDADE FEDERAL - DEPARTAMENTO DE CIENCIA DA COMPUTACAO", (70, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.line(doc, (70, 150), (doc_w - 70, 150), (200, 50, 50), 2)

    # Section 1: Introduction text lines
    cv2.putText(doc, "1. INTRODUCAO E MOTIVACAO", (70, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    text_paragraphs = [
        "A transformacao de perspectiva e uma ferramenta fundamental em Visao Computacional.",
        "Ela permite corrigir distorcoes geometricas causadas pela inclinacao da camera.",
        "Em aplicacoes de digitalizacao de documentos, a homografia planar projeta a imagem",
        "capturada em um plano bidimensional retificado, facilitando a leitura por OCR.",
        "",
        "2. FORMULACAO MATEMATICA DA HOMOGRAFIA",
        "Dada uma matriz H de dimensao 3x3 com 8 graus de liberdade, a relacao entre",
        "os pontos de origem (p) e pontos de destino (p') em coordenadas homogeneas e:",
        "                          p' = H * p",
        "Onde sao necessarios no minimo 4 pares de pontos nao-colineares para resolver H.",
        "",
        "3. EXPERIMENTOS E RESULTADOS",
        "Abaixo e apresentada a tabela de testes comparativos de precisao de canto:"
    ]

    y_offset = 240
    for line in text_paragraphs:
        if line.startswith("2.") or line.startswith("3."):
            cv2.putText(doc, line, (70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_offset += 35
        elif line.startswith("   "):
            cv2.putText(doc, line, (70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 30, 30), 1)
            y_offset += 28
        else:
            cv2.putText(doc, line, (70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
            y_offset += 25

    # Draw a table mock
    table_top = 620
    cv2.rectangle(doc, (70, table_top), (doc_w - 70, table_top + 160), (0, 0, 0), 2)
    cv2.line(doc, (70, table_top + 40), (doc_w - 70, table_top + 40), (0, 0, 0), 2)
    cv2.line(doc, (300, table_top), (300, table_top + 160), (0, 0, 0), 1)
    cv2.line(doc, (550, table_top), (550, table_top + 160), (0, 0, 0), 1)

    cv2.putText(doc, "Metodo", (100, table_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(doc, "Erro (pixels)", (330, table_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(doc, "Tempo (ms)", (580, table_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.putText(doc, "Manual (Mouse)", (100, table_top + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(doc, "0.42 px", (330, table_top + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(doc, "12.5 ms", (580, table_top + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    cv2.putText(doc, "Auto (Canny+DP)", (100, table_top + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(doc, "1.18 px", (330, table_top + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(doc, "4.2 ms", (580, table_top + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    # Draw a seal / stamp
    cv2.circle(doc, (doc_w - 150, doc_h - 180), 55, (30, 30, 180), 3)
    cv2.circle(doc, (doc_w - 150, doc_h - 180), 48, (30, 30, 180), 1)
    cv2.putText(doc, "APROVADO", (doc_w - 195, doc_h - 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 180), 2)
    cv2.putText(doc, "2026-10-04", (doc_w - 190, doc_h - 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 180), 1)

    # 2. Create a background scene (wooden table effect)
    bg_w, bg_h = 1280, 960
    bg = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
    # Give background dark wood / desk gradient texture
    for y in range(bg_h):
        val = int(35 + 20 * (y / bg_h))
        bg[y, :] = (val, val + 10, val + 25)
    
    # Add random wood noise texture
    np.random.seed(42)
    noise = np.random.randint(-10, 10, (bg_h, bg_w, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 3. Apply Perspective Transformation to simulate a document lying tilted on a table
    src_corners = np.float32([
        [0, 0],
        [doc_w - 1, 0],
        [doc_w - 1, doc_h - 1],
        [0, doc_h - 1]
    ])

    # Target points on the background canvas (tilted document)
    dst_corners = np.float32([
        [240, 140],    # Top-Left
        [980, 90],     # Top-Right
        [1120, 840],   # Bottom-Right
        [150, 780]     # Bottom-Left
    ])

    H_sim = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped_doc = cv2.warpPerspective(doc, H_sim, (bg_w, bg_h), borderValue=(0, 0, 0))

    # Mask to combine background and document smoothly
    mask = np.zeros((doc_h, doc_w), dtype=np.uint8) * 255
    cv2.rectangle(mask, (0, 0), (doc_w, doc_h), 255, -1)
    warped_mask = cv2.warpPerspective(mask, H_sim, (bg_w, bg_h))

    final_image = bg.copy()
    idx = warped_mask > 0
    final_image[idx] = warped_doc[idx]

    # Save output
    cv2.imwrite(output_path, final_image)
    print(f"Sample document generated successfully at '{output_path}'")

if __name__ == "__main__":
    generate_sample_document()
