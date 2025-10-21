import math
import matplotlib.pyplot as plt

# Constants - used for printing to stdout in color
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}


class Tester:
    """
    A test harness for evaluating price prediction models.
    
    This class provides a comprehensive testing framework that evaluates any model
    against a subset of test data and displays results in a visually satisfying way.
    
    Usage:
        def my_prediction_function(item):
            # my code here
            return my_estimate
        
        Tester.test(my_prediction_function)
    """

    def __init__(self, predictor, title=None, data=None, size=250):
        """
        Initialize the Tester.
        
        Args:
            predictor: Function that takes an item and returns a price prediction
            title: Custom title for the test (defaults to function name)
            data: Test dataset (defaults to global test data)
            size: Number of test items to evaluate (default: 250)
        """
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []

    def color_for(self, error, truth):
        """
        Determine color coding for error visualization.
        
        Args:
            error: Absolute error value
            truth: Actual price value
            
        Returns:
            Color string: "green", "orange", or "red"
        """
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"
    
    def run_datapoint(self, i):
        """
        Run prediction on a single datapoint and record results.
        
        Args:
            i: Index of the datapoint in the test data
        """
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        error = abs(guess - truth)
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40] + "..."
        
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    def chart(self, title):
        """
        Create a scatter plot visualization of predictions vs actual prices.
        
        Args:
            title: Chart title
        """
        max_error = max(self.errors)
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        """
        Generate and display a comprehensive test report with metrics and visualization.
        """
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color == "green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        """
        Run the complete test evaluation on all test items.
        """
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data=None, size=250):
        """
        Convenience method to run a test on a prediction function.
        
        Args:
            function: Prediction function to test
            data: Test dataset (optional)
            size: Number of test items (default: 250)
        """
        cls(function, data=data, size=size).run()
