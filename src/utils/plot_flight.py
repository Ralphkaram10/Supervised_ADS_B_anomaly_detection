import matplotlib.pyplot as plt
import pandas as pd

def plot_flight(path,feature='12'):
    '''
    Plot the specified feature of a file of meta messages corresponding to one flight
    x axis: time (feature '7_8')
    y axis: the specified feature
    '''
    df=pd.read_csv(path)
  
    # Create a scatter plot with different colors based on the '24' feature
    plt.scatter(df[df['24'] == 0]['7_8'], df[df['24'] == 0][feature], color='green', label='normal')
    plt.scatter(df[df['24'] == 1]['7_8'], df[df['24'] == 1][feature], color='red', label='attacked')

    # Set labels for the axes
    plt.xlabel('7_8 (seconds)')
    plt.ylabel(feature)

    # Set a title for the plot (optional)
    plt.title(f'Scatter Plot: 7_8 vs {feature} with Color Differentiation')

    # Show a legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__=='__main__':
    path="../../datasets/dataset_example_containing_one_flight_only/callsign_AAL2434_filtered_post.csv"
    plot_flight(path,feature='12')