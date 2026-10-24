from paddleocr import PaddleOCR

# Inicializa o OCR com as mesmas configurações do seu treino
ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)

img_path = '/user/home/b.nabais/flow_grpo/wandb/latest-run/files/media/images/images_39_61f9c4485ec06b053b1e.jpg'
result = ocr.ocr(img_path, cls=False)

print("\n=== Resultado do OCR ===")
if result and result[0]:
    for line in result[0]:
        texto = line[1][0]
        confianca = line[1][1]
        print(f"Detetado: '{texto}' | Confiança: {confianca:.2f}")
else:
    print("Nenhum texto detetado na imagem.")