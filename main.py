


class LanguageModel:
    def __init__(self):
        pass

    def train(self, train_data):
        print("--Training the language model...--")
        pass

    def solve_math_problem(self, question):
        print("--Solving math problem--")
        print("Question: ", question)
        dummy_solution = "2 + 2 = 4"
        print("Solution: ", dummy_solution)
        return dummy_solution



# Function to read data
def read_data(file_path):
    #Dummy data
    data = [{"problem": "How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?",
    "solution":"The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$. Therefore, the graph has $\boxed{2}$ vertical asymptotes."}]
    return data



def train_model(train_data):
    model = LanguageModel()
    model.train(train_data)
    return model


# Function to run the model and evaluate
def run_and_evaluate_model(model, test_data):
    correct = 0
    total = 0
    for data_point in test_data:
        problem = data_point['problem']
        actual_solution = data_point['solution']

        # Run model
        predicted_solution = model.solve_math_problem(problem)

        # Evaluate accuracy
        if predicted_solution == actual_solution:
            correct += 1
        total += 1

    accuracy = correct / total
    creativity = "tbd"

    return accuracy, creativity


def main():

    # Train-test data
    train_data = read_data('data/MATH/train/algebra/1.json')
    test_data = read_data('data/MATH/test/algebra/1.json')

    # Train model
    trained_model = train_model(train_data)

    # Run and evaluate model
    accuracy, creativity = run_and_evaluate_model(trained_model, test_data)

    print("Accuracy:", accuracy)
    print("Creativity:", creativity)





if __name__ == "__main__":
    main()