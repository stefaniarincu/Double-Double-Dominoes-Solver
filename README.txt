1. Librarii necesare si versiuni

numpy==1.26.2
opencv_python==4.8.1.78
cv2==4.8.1


2. Cum sa rulam si unde sunt output-urile

Toate task-urile rulează pe baza aceluiași script: main.py

Eu am rulat programul in Pycharm si am avut toate folder-ele incluse. De aceea, calea se rezuma la mentionarea fisierului necesar.


OUTPUT (pentru intreg exercitiul):
    SOL_FOLDER_PATH = 'evaluare/fisiere_solutie/332_Rincu_Stefania'

    Am declarat constanta SOL_FOLDER PATH care primeste calea de mai sus si creeaza folderele daca acestea nu exista.
    Pentru a gasi filele ".txt" care contin outputul trebuie sa cautati in "evaluare/fisiere_solutie/332_Rincu_Stefania".
    Fiecare fila este denumita dupa modelul din tema.
    Este folosita in for-urile care realizeaza iteratia prin jocuri.


Date auxiliare:
    Am introdus folder-ul "templates" pe care l-am incarcat si care contine template-urile mele generate.
    Am introdus folder-ul "aux_img" pe care l-am incarcat si care contine o imagine cu prima tabla de joc, fara piese.
    Este necesar sa descarcati aceste foldere si sa specificati calea catre ele. Aceasta poate fi modificata la inceputul scriptului.


INPUT (pentru intreg exercitiul):
    Am declarat constanta IMGS_FOLDER_PATH care poate fi modificata pentru a contine calea folder-ului de unde se preiau mutarile si imaginile pentru care rulam programul.
    Trebuie modificat dupa caz si se afla la linia 10 in script.
    Este folosita in for-urile care realizeaza iteratia prin jocuri.
    !!! Daca este modificata calea, nu adaugati "/" la final, deoarece in program il concatenez.
    IMGS_FOLDER_PATH = 'testare'


    Pentru functia de template matching am adaugat un folder care contine template-urile mele.
    Folosesc aceasta cale in functia match_number.
    !!! Daca este modificata calea, nu adaugati "/" la final, deoarece in program il concatenez si nu scoateti folder-ul de template-uri.
    TEMPLATES_FOLDER_PATH = 'templates'

    Am inclus o singura imagine auxiliara, si anume tabla de joc fara piese, taiata si redimensionata. 
    Pe aceasta o preiau la inceputul rularii script-ului.
    !!! Daca este modificata calea, nu scoateti folder-ul cu imaginea auxiliara sau imaginea cu prima tabla.
    AUX_IMAGE_FOLDER_PATH = 'aux_img/empty_table.jpg'


Functii:
    - determine_corners(contours, border) - intoarce o lista a coordonatelor pentru cel mai mare contur din cele primite
    - detect_entire_board(original_image) - detecteaza tabla mare si intoarce matricea acesteia; apeleaza functia determine_corners
    - extract_DDD_table(original_image) - preia careul de joc si il returneaza; apeleaza functia detect_entire_board si determine_corners
    - determine_interpolation(first_table, aux_table) - pentru doua imagini realizeaza o interpolare pentru intensitatea pixelilor, returneaza o matrice (harta de mappare)
    - determine_piece_coordinates(just_game_table, aux_table_gray, game_matrix, game_round) - determina coordonatele unei piese si le returneaza
    - match_number(patch) - realizeaza template matching; este necesar TEMPLATES_FOLDER_PATH
    - determine_number_on_piece(just_game_table, piece_edges) - preia piesa si apeleaza match_number pentru a o dteremina; returneaza o cifra
    - calculate_score(players_pos, current_player, coord, nr_on_piece) - calculeaza scorul