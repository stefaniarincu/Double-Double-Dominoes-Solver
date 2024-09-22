import cv2 as cv
import numpy as np
import os

# Path to the folder where solutions will be saved
SOL_FOLDER_PATH = 'evaluare/fisiere_solutie/332_Rincu_Stefania'
os.makedirs(SOL_FOLDER_PATH, exist_ok=True)

# Path to the folder from where we will retrieve the moves and images for the program
# Do not add a "/" at the end
IMGS_FOLDER_PATH = 'testare'

# Path to retrieve the templates for template matching
# The templates have been loaded into the "templates" folder, which was provided with the assignment
# Do not add a "/" at the end
TEMPLATES_FOLDER_PATH = 'templates'

# Path to retrieve the auxiliary game board image
# The auxiliary image has been loaded into the "aux_img" folder, which was provided with the assignment
AUX_IMAGE_FOLDER_PATH = 'aux_img/empty_table.jpg'


# Matrix that stores the score values on the edge of the game board
outside_points = [-1, 1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0, 3, 5, 4, 1, 6, 2, 4, 5, 5, 0, 6, 3, 4,
                  2, 0, 1, 5, 1, 3, 4, 4, 4, 5, 0, 6, 3, 5, 4, 1, 3, 2, 0, 0, 1, 1, 2, 3, 6, 3, 5, 2, 1,
                  0, 6, 6, 5, 2, 1, 2, 5, 0, 3, 3, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4, 2, 6, 2, 3, 1, 6, 5,
                  6, 2, 0, 4, 0, 1, 6, 4, 4, 1, 6, 6, 3, 0]

# Matrix that stores the bonus points on the game board
table_points = [[5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
                [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
                [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
                [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
                [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
                [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
                [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
                [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
                [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
                [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5]]

# Simulate the grid (intersection of neighboring rows and columns determines the cells on the game board)
lines_horizontal = [[(0, i), (1499, i)] for i in range(0, 1501, 100)]
lines_vertical = [[(i, 0), (i, 1499)] for i in range(0, 1501, 100)]


# Auxiliary function to display images
def show_image(title, img):
    img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function that determines the corners of the game board based on a set of contours
def determine_corners(contours, border):
    max_area = 0

    for i in range(len(contours)):
        # Ensure that the contour has more than 3 points (to form a quadrilateral)
        if len(contours[i]) > 3:
            possib_top_left = None
            possib_bottom_right = None

            for point in contours[i].squeeze():
                # Find the top-left corner (minimizing x + y coordinates)
                if possib_top_left is None or point[0] + point[1] < possib_top_left[0] + possib_top_left[1]:
                    possib_top_left = point

                # Find the bottom-right corner (maximizing x + y coordinates)
                if possib_bottom_right is None or point[0] + point[1] > possib_bottom_right[0] + possib_bottom_right[1]:
                    possib_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]

            # Calculate the area of the current contour formed by the four points and compare it to the maximum area 
            if cv.contourArea(np.array([[possib_top_left], [possible_top_right], [possib_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possib_top_left], [possible_top_right], [possib_bottom_right], [possible_bottom_left]]))

                # Adjust the corners by adding or subtracting a specified value
                # Because of dilation, the corners to be detected further outside
                top_left = [possib_top_left[0] + border, possib_top_left[1] + border]
                bottom_right = [possib_bottom_right[0] - border, possib_bottom_right[1] - border]
                top_right = [possible_top_right[0] - border, possible_top_right[1] + border]
                bottom_left = [possible_bottom_left[0] + border, possible_bottom_left[1] - border]

    return [top_left, top_right, bottom_right, bottom_left]


# Function that detects the entire game board (the one placed on the table)
def detect_entire_board(original_image):
    # Remove the dark brown color, which represents the table
    original_image_hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)  
    mask_table_hsv = cv.inRange(original_image_hsv, np.array([24, 0, 0]), np.array([255, 255, 255]))  # Mask for everything except dark brown
    result_image = cv.bitwise_and(original_image_hsv, original_image_hsv, mask=mask_table_hsv)  
    result_image_gray = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)  

    # Remove noise, smooth the image, and enhance contrasts
    image_m_blur = cv.medianBlur(result_image_gray, 3)  
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0) 

    _, thresh = cv.threshold(image_sharpened, 10, 255, cv.THRESH_BINARY)  

    # Detect edges using the Canny edge detector
    edges = cv.Canny(thresh, 200, 400)  

    # The detected edges are thin and segmented -> connect and thicken them using dilation
    kernel = np.ones((5, 5), np.uint8)  
    image_dilated = cv.dilate(edges, kernel, iterations=6)  

    contours, _ = cv.findContours(image_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  
    width, height, _ = original_image.shape  

    # Get the corners of the detected board and create a matrix for perspective transformation
    dd_entire_table = np.array(determine_corners(contours, 40), dtype="float32") 
    dd_destination_table = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")  

    matrix = cv.getPerspectiveTransform(dd_entire_table, dd_destination_table)  
    entire_board = cv.warpPerspective(original_image, matrix, (width, height))  

    return entire_board 


# Function that detects the inner game board
def extract_DDD_table(original_image):
    # First, detect the entire game board and crop a portion of the edges to remove the score table
    entire_board = detect_entire_board(original_image)
    entire_board = entire_board[250:-250, 250:-250, :]  

    entire_board_hsv = cv.cvtColor(entire_board, cv.COLOR_BGR2HSV)  

    # Reduce noise and smooth the image to highlight the edges of the inner game board
    # In some images, game pieces are placed near the edges, so without this preprocessing step the board 
    # might not be detected accurately or some pieces might be cut off, which disrupt the template matching step
    entire_board_m_blur = cv.medianBlur(entire_board_hsv, 21) 
    entire_board_g_blur = cv.GaussianBlur(entire_board_m_blur, (0, 0), 5) 

    # Keep only colors in the green-blue range (which represents the color of the board)  
    mask_table_hsv = cv.inRange(entire_board_g_blur, np.array([30, 0, 0]), np.array([134, 255, 255]))
    result_image = cv.bitwise_and(entire_board_g_blur, entire_board_g_blur, mask=mask_table_hsv)
    result_image_gray = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)  

    # Remove noise, smooth the image, and enhance contrasts
    image_m_blur = cv.medianBlur(result_image_gray, 5) 
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.6, 0) 

    _, thresh = cv.threshold(image_sharpened, 5, 255, cv.THRESH_BINARY)  

    # Detect edges using the Canny edge detector
    edges = cv.Canny(thresh, 200, 400)  

    # The detected edges are thin and segmented, so they are connected and thickened using dilation
    # Apply a median filter to separate unwanted sections that join after dilation
    kernel = np.ones((5, 5), np.uint8) 
    img_dilated = cv.dilate(edges, kernel, iterations=2) 
    img_m_blur = cv.medianBlur(img_dilated, 5) 

    contours, _ = cv.findContours(img_m_blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    width = 1500
    height = 1500

    # Get the corners of the detected game board and create a matrix for perspective transformation
    dd_table = np.array(determine_corners(contours, 5), dtype="float32")  
    dd_destination_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")  

    matrix = cv.getPerspectiveTransform(dd_table, dd_destination_puzzle) 
    just_game_table = cv.warpPerspective(entire_board, matrix, (width, height)) 

    return just_game_table  


# Function that determines the interpolation between the empty board image and the first board of a new game
def determine_interpolation(first_table, aux_table):
    first_table_gray = cv.cvtColor(first_table, cv.COLOR_BGR2GRAY)
    aux_table_gray = cv.cvtColor(aux_table, cv.COLOR_BGR2GRAY)

    # 1. Compute the histogram of pixel intensities for each image using 256 bins
    #    Each bin represents one possible value (ranging from 0 to 255 inclusive)
    # 2. Normalize the histogram to the range (0, 1) to determine its cumulative distribution
    # 3. Calculate the cumulative sum of the normalized histogram

    first_table_histogram, _ = np.histogram(first_table_gray.flatten(), bins=256, range=(0, 256))
    first_table_hist_norm = first_table_histogram / 2250000 
    current_cumsum = first_table_hist_norm.cumsum() 

    aux_table_histogram, _ = np.histogram(aux_table_gray.flatten(), bins=256, range=(0, 256))
    aux_table_hist_norm = aux_table_histogram / 2250000  
    needed_cumsum = aux_table_hist_norm.cumsum()  

    # Determine the linear interpolation between the current cumulative sum (for the first board of this game)
    # and the target cumulative sum (for the auxiliary empty board) to adjust pixel intensities
    interpolation = np.interp(current_cumsum, needed_cumsum, range(256))

    return interpolation.astype(np.uint8) 


# Function that determines the position of a game piece placed in a given round
def determine_piece_coordinates(just_game_table, aux_table_gray, game_matrix):
    just_game_table_gray = cv.cvtColor(just_game_table, cv.COLOR_BGR2GRAY)

    # Apply median blur to both images to reduce noise
    game_table_m_blur = cv.medianBlur(just_game_table_gray, 11)
    aux_table_m_blur = cv.medianBlur(aux_table_gray, 11)

    # Compute the difference between the two blurred images
    # Clip the result to ensure pixel values are within the valid range (0 to 255)
    diff = np.clip(game_table_m_blur.astype(np.int64) - aux_table_m_blur.astype(np.int64), 0, 255).astype(np.uint8)

    # Apply morphological opening to remove background noise
    diff_opened = cv.morphologyEx(diff, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=2)
    # Apply dilation to ensure any black areas in the center of the piece are included
    diff_dilated = cv.dilate(diff_opened, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))

    coord = []
    piece_edges = []

    height, width, _ = just_game_table.shape

    # Iterate through the cells of the game board
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            # Adjust coordinates by 5 pixels to better account for the piece's interior
            # This helps avoid false positives from neighboring cells
            y_min = lines_vertical[j][0][0] + 5
            y_max = lines_vertical[j + 1][1][0] - 5
            x_min = lines_horizontal[i][0][1] + 5
            x_max = lines_horizontal[i + 1][1][1] - 5

            patch = diff_dilated[x_min:x_max, y_min:y_max].copy()

            # Compute the mean of the patch; if it's greater than 29, it indicates the presence of a piece
            mean_patch = np.mean(patch)

            if mean_patch > 29 and game_matrix[i][j] == 0:
                game_matrix[i][j] = 1
                coord.append([i, j])
                # Adjust coordinates by adding or subtracting 10 to include a small border around the piece
                # This helps with template matching in case the piece is slightly misaligned
                piece_edges.append(
                    [max(y_min - 10, 0), min(y_max + 10, width), max(x_min - 10, 0), min(x_max + 10, height)])

    return coord, piece_edges


# Function that performs the Template Matching step in order to identify the number on a game piece
def match_number(patch):
    maxi = -np.inf  
    poz = -1  

    # Retrieve all template images from the templates folder (path needs to be set at the beginning of the script)
    file_list = os.listdir(TEMPLATES_FOLDER_PATH)

    # Iterate through all template files in the folder
    for file_name in file_list:
        nr = int(file_name[1])
        template = cv.imread(TEMPLATES_FOLDER_PATH + f'/{file_name}')

        # Highlight the dots and white areas of the piece
        template_hsv = cv.cvtColor(template, cv.COLOR_BGR2HSV)
        mask_table_hsv = cv.inRange(template_hsv, np.array([82, 0, 0]), np.array([255, 108, 255]))
        result_template = cv.bitwise_and(template_hsv, template_hsv, mask=mask_table_hsv)
        template_gray = cv.cvtColor(result_template, cv.COLOR_BGR2GRAY)

        # Perform template matching using normalized correlation coefficient method
        corr = cv.matchTemplate(patch, template_gray, cv.TM_CCOEFF_NORMED)
        corr = np.max(corr) 

        # Update the number if the current template has the highest correlation with the patch
        if corr > maxi:
            maxi = corr
            poz = nr

    return poz


# Function that extracts a patch representing half of a game piece and preprocesses it 
def determine_number_on_piece(just_game_table, piece_edges):
    patch = just_game_table[piece_edges[2]:piece_edges[3], piece_edges[0]:piece_edges[1]].copy()

    # Highlight the dots and white areas of the piece, and possibly remove blue from the board if necessary
    patch_hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
    mask_table_hsv = cv.inRange(patch_hsv, np.array([82, 0, 0]), np.array([255, 108, 255]))
    result_image = cv.bitwise_and(patch_hsv, patch_hsv, mask=mask_table_hsv)
    patch_gray = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)

    # Remove noise and erode the image to eliminate the black part (there were cases 
    # where the line in the middle of the piece was mistakenly identified as the number 6)
    kernel = np.ones((3, 3), np.uint8)
    patch_m_blur = cv.medianBlur(patch_gray, 7)
    patch_eroded = cv.erode(patch_m_blur, kernel)

    _, thresh = cv.threshold(patch_eroded, 80, 255, cv.THRESH_BINARY)

    # Apply morphological closing to fill in gaps and blur the result to improve template matching
    patch_morph_close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    patch_g_blur = cv.GaussianBlur(patch_morph_close, (5, 5), 4)

    # Perform template matching to identify the number on the piece
    return match_number(patch_g_blur)


# Function to calculate the score based on the current game state
def calculate_score(players_pos, current_player, coord, nr_on_piece):
    # Calculate the points based on the table points for the given coordinates
    points = table_points[coord[0][0]][coord[0][1]] + table_points[coord[1][0]][coord[1][1]]

    # Double the points if the numbers on the piece are the same
    if nr_on_piece[0] == nr_on_piece[1]:
        points *= 2

    # Add points if the number on the piece matches the number in the outside_points list for the current player
    if (outside_points[players_pos[current_player - 1][-1]] == nr_on_piece[0]
            or outside_points[players_pos[current_player - 1][-1]] == nr_on_piece[1]):
        points += 3

    # Update the score for the current player if points are positive
    if points > 0:
        players_pos[current_player - 1].append(players_pos[current_player - 1][-1] + points)

    # Update the score for the opponent if the number on the piece matches the opponent's outside_points number
    if (outside_points[players_pos[(current_player - 2) % 2][-1]] == nr_on_piece[0]
            or outside_points[players_pos[(current_player - 2) % 2][-1]] == nr_on_piece[1]):
        players_pos[(current_player - 2) % 2].append(players_pos[(current_player - 2) % 2][-1] + 3)

    return points


# Load the auxiliary empty table image
# If needed, modify the path at the beginning of the script
aux_table = cv.imread(AUX_IMAGE_FOLDER_PATH)
aux_table_gray = cv.cvtColor(aux_table, cv.COLOR_BGR2GRAY)

for i in range(1, 6):
    # Initialize the game table and player positions
    game_table = [[0 for _ in range(15)] for _ in range(15)]
    players_positions = [[0], [0]]
    cnt = 1

    # Read the moves for the current game from the text file
    with open(f'{IMGS_FOLDER_PATH}/{i}_mutari.txt', 'r') as whose_turn:
        for line in whose_turn:
            line = line.strip()

            if line != "":
                path, player = line.split(" ")

                # Load and process the current image
                image = cv.imread(f'{IMGS_FOLDER_PATH}/{path}')
                print(f'Processing image --- {path} ---')

                # Crop the top and bottom portions of the image
                image_truncated = image[400:3680, :, :]

                # Extract the game board from the image
                just_board_game = extract_DDD_table(image_truncated)

                # Determine the interpolation matrix only for the first game board
                if cnt == 1:
                    interpolation_matrix = determine_interpolation(just_board_game, aux_table)

                # Apply the Look Up Table (LUT) transformation to adjust the intensities
                just_board_game = cv.LUT(just_board_game, interpolation_matrix)

                # Determine the coordinates and edges of the game pieces
                coordinates, pieces_edges = determine_piece_coordinates(just_board_game, aux_table_gray, game_table)

                k = 0 # This variable counts the number of grids that appear to be occupied in one round
                coord_entire_piece = []
                numbers_piece = []

                # Write the results to the output file
                with open(SOL_FOLDER_PATH + f'/{path[:-4]}.txt', 'w') as output_file:
                    while k < 2 and k < len(coordinates):
                        col = chr(ord('A') + coordinates[k][1])
                        nr_on_piece = determine_number_on_piece(just_board_game, pieces_edges[k])

                        output_file.write(f"{coordinates[k][0] + 1}{col} {nr_on_piece}\n")

                        coord_entire_piece.append(coordinates[k])
                        numbers_piece.append(nr_on_piece)

                        k += 1

                    # Calculate and write the score if two grids were processed
                    if k == 2:
                        points_round = calculate_score(players_positions, int(player[-1]), coord_entire_piece, numbers_piece)
                        output_file.write(f"{points_round}\n")
                    else:
                        # Write placeholder values if fewer than two grids were processed
                        while k < 2:
                            output_file.write(f"<3 0\n")
                            k += 1

                        output_file.write(f"0\n")

            cnt += 1