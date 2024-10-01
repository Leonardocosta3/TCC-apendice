#Importação das Bibliotecas
import cv2
import time
import pandas as pd
from ultralytics import YOLOv10

# Carregar o modelo YOLOv10
model = YOLOv10(f'Yolov10s\weights\Yolov10Sbest.pt')

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0) # Indice da cámera

# Variáveis para o contador e estado dos olhos
contador_piscadas = 0 # Variavel de contagem de Piscadas
piscadas_tempo = 0 # Variavel de contagem de duração de uma Piscada
estado_anterior_olhos = "aberto"
limite_contador_olhos = 30 # Limite de piscadas em um determinado periodo
tempo_piscada = 0

# Variáveis para o contador e estado da cabeça
contador_cabeca_baixa = 0 # Variavel de contagem de Pescadas
tempo_cabeça_baixa = 0 # Variavel de contagem de duração de uma Pescada
estado_anterior_cabeca = "atenta"
limite_contador_cabeca = 5 # Limite de pescadas em um determinado periodo

# Variáveis para controlar o alerta
alerta_ativo = False
inicio_alerta = 0
duracao_alerta = 10  # Duração do alerta em segundos

# Loop de avaliação de sonolência
start_loop_time = time.time()  # Início do loop de 60 segundos
loop_duration = 60  # Duração do loop do periodo de avaliação de sono

# Variáveis para contagem de tempo com cabeça baixa e olhos fechados
start_time = 0
total_time = 0
detectando_cabeca = False
detectando_olhos = False
limite_tempo_olhos = 2 # Tempo máximo de olhos fechados para gerar o alerta
limite_tempo_cabeca = 3 # Tempo máximo com a cabeça baixa para gerar o alerta

# Variáveis para controle de Bocejo
detectando_bocejo = False
temp_bocejo = 5  # Intervalo de tempo para considerar um novo bocejo
intervalo_entre_bocejos = 20  # Intervalo de tempo para a próxima detecção após contar um bocejo

# Variáveis para contagem de bocejos
start_time_bocejo = 0
cont_bocejo = 0
ultimo_bocejo_time = 0

# Criação do DataFrame
df = pd.DataFrame(columns=["Arquivo", "Evento", "Data", "Hora", "Duracao", "Nivel"])

# Inicialização da captura de vídeo
while cap.isOpened():
        ret, frame = cap.read()
        results = model(frame)

        # Define a variavel das caixas delimitadoras
        annotated_frame = results[0].plot()

        # Inicia o estado atual como estado aberto
        estado_atual_olhos = estado_anterior_olhos

        # Inicia o estado atual como estado aberto
        estado_atual_cabeca = estado_anterior_cabeca

        # Variável com o status de detecção dos olhos
        classe_detectada = False

        # Variável com o status de detecção da cabeça
        classe_detectada_cabeca = False

        # Variável que verifica bocejo
        classe_detectada_bocejo = False

        # Loop com os resultados das detecções
        for obj in results:
            nomes = obj.names
            for item in obj.boxes:
                x1, y1, x2, y2 = item.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(item.cls[0])
                nomeClasse = nomes[cls]
                conf = round(float(item.conf[0]), 2)

                # Lógica para capturar quando há uma piscada
                if nomeClasse == "olhosFechados":
                    estado_atual_olhos = "fechado"
                    # Registra o início da piscada
                    if estado_anterior_olhos == "aberto":
                        tempo_inicio_piscada = time.time()
                elif nomeClasse == 'olhosAbertos' or nomeClasse == 'sonolento':
                    estado_atual_olhos = "aberto"
                    # Registra o fim da piscada
                    if estado_anterior_olhos == "fechado":
                        tempo_fim_piscada = time.time()
                        duracao_piscada = tempo_fim_piscada - tempo_inicio_piscada
                        tempo_piscada = duracao_piscada
                        # Define o Nivel de Fádiga dos olhos
                        if tempo_piscada >= 0.5 and tempo_piscada <=1:
                            nivel_fadiga_olhos = "Leve"
                        elif tempo_piscada > 1 and tempo_piscada <=1.5:
                            nivel_fadiga_olhos = "Media"
                        elif tempo_piscada >1.5 and tempo_piscada <=2:
                            nivel_fadiga_olhos = "Forte"
                        elif tempo_piscada >2:
                            nivel_fadiga_olhos = "Dormiu"
                
                # Verificar se a detecção é da classe "olhosFechados"
                if nomeClasse == "olhosFechados":
                    classe_detectada = True
                    if not detectando_olhos:
                        start_time = time.time()  # Inicia o contador de tempo
                        detectando_olhos = True
                
                # Lógica para capturar cabeça baixa
                if nomeClasse == "cabecaBaixa":
                    estado_atual_cabeca = "baixa"
                    # Registra o início da pescada
                    if estado_anterior_cabeca == "atenta":
                        tempo_inicio_pescada = time.time()
                elif nomeClasse == 'atento':
                    estado_atual_cabeca = "atenta"
                    # Registra o fim da pescada
                    if estado_anterior_cabeca == "baixa":
                        tempo_fim_pescada = time.time()
                        duracao_pescada = tempo_fim_pescada - tempo_inicio_pescada
                        tempo_pescada = duracao_pescada
                        # Define o Nivel de Fadiga da cabeça
                        if tempo_pescada <= 2:
                            nivel_fadiga_cabeca = "Leve"
                        elif tempo_pescada > 2 and tempo_pescada <= 3:
                            nivel_fadiga_cabeca = "Forte"
                        elif tempo_pescada > 3:
                            nivel_fadiga_cabeca = "Dormiu"
                
                # Verificar se a detecção é da classe "cabecaBaixa"
                if nomeClasse == "cabecaBaixa":
                    classe_detectada_cabeca = True
                    if not detectando_cabeca:
                        start_time = time.time()  # Inicia o contador de tempo
                        detectando_cabeca = True
                
                # Verificar se a detecção é da classe "Bocejando"
                if nomeClasse == "bocejando":
                    classe_detectada_bocejo = True
                    if not detectando_bocejo:
                        start_time_bocejo = time.time()  # Inicia o contador de tempo
                        detectando_bocejo = True

        # Verificar se 60 segundos se passaram               
        elapsed_time = time.time() - start_loop_time
        if elapsed_time >= loop_duration:
            # Se o contador de piscadas atingir o limite, exibe o alerta
            if piscadas_tempo >= limite_contador_olhos:
                alerta_ativo = True
                inicio_alerta = time.time()

            # Reiniciar o contador e o tempo para o próximo ciclo de 60 segundos
            piscadas_tempo = 0
            start_loop_time = time.time()

            # Se o contador de pescadas atingir o limite, exibe o alerta
            if tempo_cabeça_baixa >=limite_contador_cabeca:
                alerta_ativo = True
                inicio_alerta = time.time()

            # Reiniciar o contador e o tempo para o próximo ciclo de 60 segundos
            tempo_cabeça_baixa = 0
            start_loop_time = time.time()
            
        # Exibir o alerta por 10 segundos
        if alerta_ativo:
            tempo_alerta = time.time() - inicio_alerta
            if tempo_alerta < duracao_alerta:
                # Desenhar o alerta
                cv2.rectangle(annotated_frame, (175, 420), (455, 460), (0, 0, 255), -1)
                cv2.putText(annotated_frame, f"ALERTA DE SONOLENCIA!", (180, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                alerta_ativo = False  # Desativa o alerta após 10 segundos

        # Incrementar os contadores se houve uma piscada
        if estado_atual_olhos == "aberto" and estado_anterior_olhos == "fechado":
            contador_piscadas += 1
            piscadas_tempo += 1
            nova_linha = pd.DataFrame([{"Arquivo": "teste02", "Evento": "Piscada detectada", 
                                    "Data": time.strftime("%d-%m-%Y"), "Hora": time.strftime("%H:%M:%S"), 
                                    "Duracao": f"{tempo_piscada:.2f}","Nivel": nivel_fadiga_olhos }])
            df = pd.concat([df, nova_linha], ignore_index=True)
           
        # Atualizar o estado anterior
        estado_anterior_olhos = estado_atual_olhos

        # Incrementar os contadores se houve uma pescada
        if estado_atual_cabeca == "atenta" and estado_anterior_cabeca == "baixa":
            contador_cabeca_baixa += 1
            tempo_cabeça_baixa += 1
            nova_linha = pd.DataFrame([{"Arquivo": "teste02", "Evento": "Cabeca baixa detectada", 
                                    "Data": time.strftime("%d-%m-%Y"), "Hora": time.strftime("%H:%M:%S"), 
                                    "Duracao": f"{tempo_pescada:.2f}", "Nivel": nivel_fadiga_cabeca }])
            df = pd.concat([df, nova_linha], ignore_index=True)

        # Atualizar o estado anterior
        estado_anterior_cabeca = estado_atual_cabeca

        # Se a classe "olhosFechados" foi detectada, calcula o tempo total
        if classe_detectada and detectando_olhos:
            total_time = time.time() - start_time
            if total_time >= limite_tempo_olhos:
                cv2.rectangle(annotated_frame, (210, 210), (430, 250), (0, 0, 255), -1)
                cv2.putText(annotated_frame, "ALERTA: DORMINDO", (215, 238),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                '''nova_linha = pd.DataFrame([{"Arquivo": "teste02", "Evento": "Olhos Fechados detectados", 
                                        "Data": time.strftime("%d-%m-%Y"), "Hora": time.strftime("%H:%M:%S"), 
                                        "Duracao": f"{total_time:.2f}"}])
                df = pd.concat([df, nova_linha], ignore_index=True)'''
        elif not classe_detectada and detectando_olhos:
            detectando_olhos = False  # Para o tempo quando a classe não está mais sendo detectada
            total_time = 0

        # Se a classe "cabecaBaixa" foi detectada, calcula o tempo total
        elif classe_detectada_cabeca and detectando_cabeca:
            total_time = time.time() - start_time
            if total_time >= limite_tempo_cabeca:
                cv2.rectangle(annotated_frame, (210, 210), (430, 250), (0, 0, 255), -1)
                cv2.putText(annotated_frame, "ALERTA: DORMINDO", (215, 238),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                '''nova_linha = pd.DataFrame([{"Arquivo": "teste02", "Evento": "Cabeca Baixa detectada", 
                                        "Data": time.strftime("%d-%m-%Y"), "Hora": time.strftime("%H:%M:%S"), 
                                        "Duracao": f"{total_time:.2f}", "Nivel": "Dormiu"}])
                df = pd.concat([df, nova_linha], ignore_index=True)'''
        elif not classe_detectada_cabeca and detectando_cabeca:
            detectando_cabeca = False  # Para o tempo quando a classe não está mais sendo detectada
            total_time = 0

        # Se a classe "bocejando" foi detectada e estamos detectando bocejo
        if classe_detectada_bocejo and detectando_bocejo:
            if time.time() - ultimo_bocejo_time >= intervalo_entre_bocejos:
                total_time_bocejo = time.time() - start_time_bocejo
                if total_time_bocejo >= temp_bocejo:
                    cont_bocejo += 1
                    ultimo_bocejo_time = time.time()  # Atualiza o tempo do último bocejo
                    nova_linha = pd.DataFrame([{"Arquivo": "teste02", "Evento": "Bocejo detectado", 
                                        "Data": time.strftime("%d-%m-%Y"), "Hora": time.strftime("%H:%M:%S"), 
                                        "Duracao": f"{total_time_bocejo:.2f}", "Nivel": "Leve"}])
                    df = pd.concat([df, nova_linha], ignore_index=True)
        elif not classe_detectada_bocejo:
            detectando_bocejo = False  # Reinicia o estado de detecção se bocejo não for detectado

        # Mostra o contador de piscadas
        cv2.putText(annotated_frame, f"Piscadas: {contador_piscadas}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostra o contador de bocejos
        cv2.putText(annotated_frame, f"Bocejos: {cont_bocejo}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostra o contador de tempo enquanto a classe "olhos fechados" esta sendo detectada
        if detectando_olhos == True:
            cv2.putText(annotated_frame, f"Olhos Fechados: {total_time:.2f}", (355, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Mostra o contador de tempo enquanto a classe "Cabeça Baixa" esta sendo detectada
        if detectando_cabeca == True:
            cv2.putText(annotated_frame, f"Cabeca Baixa: {total_time:.2f}", (380, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostra as caixas delimitadoras
        cv2.imshow('YoloV10', annotated_frame)

        # Sair do loop ao pressionar 'Esc'
        if cv2.waitKey(1) == 27: 
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

df.to_excel("output4.xlsx", index=False)