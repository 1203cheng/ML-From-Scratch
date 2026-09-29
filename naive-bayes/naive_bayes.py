import math

# Constant
CLASS = 10


def min_idx(list):
    min_i, min_v = 0, list[0]
    for i, v in enumerate(list):
        if v < min_v:
            min_i, min_v = i, v
    return min_i


def read_image(file):
    file.seek(4)
    num = int.from_bytes(file.read(4), 'big')
    rows = int.from_bytes(file.read(4), 'big')
    cols = int.from_bytes(file.read(4), 'big')

    return [[file.read(cols) for _ in range(rows)] for _ in range(num)]

def read_label(file):
    file.seek(4)
    num = int.from_bytes(file.read(4), 'big')

    return file.read(num)

def single_pass_mean_variance(pixel, x):
    n, mean, M2 = pixel

    n += 1
    delta = x - mean
    mean += (delta / n)
    delta2 = x - mean
    M2 += (delta2 * delta)

    pixel[0] = n
    pixel[1] = mean
    pixel[2] = M2
    return

def calculate_sd(pixel):
    n, mean, M2 = pixel
    PSEUDOCOUNT = 10

    sd = math.sqrt(M2 / (n - 1) if n > 1 else 0)
    pixel[2] = sd + PSEUDOCOUNT
    return

def prediction_discrete(prior, likelihood, image):
    # Calculate posterior
    bins = len(likelihood[0][0][0])
    level = 256 / bins
    marginal = 0
    posterior = [0] * 10
    max_post = 1
    for c in range(len(likelihood)):
        for i in range(len(likelihood[0])):
            for j in range(len(likelihood[0][0])):
                posterior[c] += likelihood[c][i][j][int(image[i][j] / level)]
        posterior[c] += prior[c]
        marginal += posterior[c]
    posterior = [e / marginal for e in posterior]

    # Output
    min_idx, min_post = 0, posterior[0]
    for idx, class_post in enumerate(posterior):
        print(f"{idx}: {class_post}")
        if class_post < min_post:
            min_idx, min_post = idx, class_post

    return min_idx

def prediction_continuous(prior, likelihood, image):
    Gaussian = lambda x, mean, sd: math.log(2 * math.pi * (sd ** 2)) / 2 + ((x - mean) ** 2) / (2 * (sd ** 2))
    marginal = 0
    posterior = [0] * 10
    max_post = 1
    for c, class_post in enumerate(likelihood):
        for i, row in enumerate(class_post):
            for j, col in enumerate(row): 
                posterior[c] += Gaussian(image[i][j], col[1], col[2])
        posterior[c] += prior[c]
        marginal += posterior[c]
    posterior = [e / marginal for e in posterior]

    # Output
    min_idx, min_post = 0, posterior[0]
    for idx, class_post in enumerate(posterior):
        print(f"{idx}: {class_post}")
        if class_post < min_post:
            min_idx, min_post = idx, class_post

    return min_idx

   
if __name__ == "__main__":
    # Mode choosing
    print("Toggle option:\n\t0: discrete mode\n\t1: continuous mode\n")
    mode = int(input("Enter 0 or 1: "))

    # Loading and parsing data
    with open('train-images.idx3-ubyte', 'rb') as file:
        train_images = read_image(file)

    with open('train-labels.idx1-ubyte', 'rb') as file:
        train_labels = read_label(file)

    with open('t10k-images.idx3-ubyte', 'rb') as file:
        test_images = read_image(file)

    with open('t10k-labels.idx1-ubyte', 'rb') as file:
        test_labels = read_label(file)

    image_rows = len(train_images[0])
    image_cols = len(train_images[0][0])
    train_num = len(train_images)
    test_num = len(test_images)

    # Calculating prior
    prior = [0 for _ in range(CLASS)]
    for label in train_labels:
        prior[label] += 1

    prior = [-math.log(e / train_num) for e in prior]

    if mode == 0:
        # Discrete Mode
        PSEUDOCOUNT = 1
        BINS = 32
        level = 256 / BINS
        likelihood = []
        for _ in range(CLASS):
            class_list = []
            for _ in range(image_rows):
                row = []
                for _ in range(image_cols):
                    row.append([PSEUDOCOUNT for _ in range(BINS)])

                class_list.append(row)
            
            likelihood.append(class_list)

        total_num = CLASS * image_rows * image_cols * BINS

        # Training data processing
        for idx, image in enumerate(train_images):
            for row in range(image_rows):
                for col, pixel in enumerate(image[row]):
                    likelihood[train_labels[idx]][row][col][int(pixel / level)] += 1

        total_num += train_num

        for c in range(CLASS):
            for i in range(image_rows):
                for j in range(image_cols):
                    for b in range(BINS):
                        likelihood[c][i][j][b] = -math.log(likelihood[c][i][j][b] / total_num)

        # Predicting test data
        error = 0
        for idx, image in enumerate(test_images):
            pred = prediction_discrete(prior, likelihood, image)
            ans = test_labels[idx]
            if pred != ans: error += 1
            print(f"Prediction: {pred}, Ans: {ans}\n")

        # Print imagination
        print('Imagination of numbers in Bayesian classifier:')
        for i in range(10): 
            print(f"\n{i}:")

            border = 128 / level
            imagination = lambda bin: '0' if bin < border else '1'
            for row in likelihood[i]:
                print(' '.join(imagination(min_idx(col)) for col in row))

        print(f"Error rate: {error / test_num}")

    else:
        # Continuous mode
        likelihood = []
        for _ in range(CLASS):
            class_list = []
            for _ in range(image_rows):
                row = []
                for _ in range(image_cols):
                    row.append([0] * 3)

                class_list.append(row)
            
            likelihood.append(class_list)
        # likelihood[class][row][col][n, mean, M2]
            
        # Training data processing
        for idx, image in enumerate(train_images):
            label = train_labels[idx]
            for i, row in enumerate(image):
                for j, col in enumerate(row):
                    single_pass_mean_variance(likelihood[label][i][j], col)

        for c in range(CLASS):
            for i in range(image_rows):
                for j in range(image_cols):
                    calculate_sd(likelihood[c][i][j])

        # Predicting test data
        error = 0
        for idx, image in enumerate(test_images):
            pred = prediction_continuous(prior, likelihood, image)
            ans = test_labels[idx]
            if pred != ans: error += 1
            print(f"Prediction: {pred}, Ans: {ans}\n")


        # Print imagination
        print('Imagination of numbers in Bayesian classifier:')
        for i in range(10): 
            print(f"\n{i}:")

            border = 128
            imagination = lambda val: '0' if val < border else '1'
            for row in likelihood[i]:
                print(' '.join(imagination(col[1]) for col in row))

        print(f"Error rate: {error / test_num}")
            
        