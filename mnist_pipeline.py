"""
KFP: kubeflow pipeline sdk 
- dsl(Domain Specific Language): DSL is SDK for defining from python code to k8s pipeline specification.

-- @dsl.component: using decorator pattern
-- dsl.InputPath/OutputPath: Type hint which define the mechanism for passing data(artifact) between components.
--- At actual runtime, KFP injects Minio(S3) and mounted local file system path(string) into the variable specified by this type.

-- @dsl.pipeline: Define DAG(Piepline), 

- kfp.compiler: Compile(Transpile) the written python pipeline object(DAG) into an Argo Workflow YAML file that k8s can understand.
"""

from kfp import dsl
from kfp import compiler

@dsl.component(base_image='python:3.9', packages_to_install=['tensorflow', 'numpy'])
def preprocess_data(
    x_train_path: dsl.OutputPath("Numpy"),
    y_train_path: dsl.OutputPath("Numpy"),
    x_test_path: dsl.OutputPath("Numpy"),
    y_test_path: dsl.OutputPath("Numpy")
):
    """
    The step of downloading data, normalizing it and serializing it for passing to the next step.
    
    :param x_train_path: File system path where the results will be saved when the function ends. 
    :type x_train_path: dsl.OutputPath("Numpy")
    :param y_train_path: File system path where the results will be saved when the function ends.
    :type y_train_path: dsl.OutputPath("Numpy")
    :param x_test_path: File system path where the results will be saved when the function ends.
    :type x_test_path: dsl.OutputPath("Numpy")
    :param y_test_path: File system path where the results will be saved when the function ends.
    :type y_test_path: dsl.OutputPath("Numpy")
    
    - Parameters action: KFP system, when container runs, creates a temporary path and allocates into the variable as string. If developer writes a file in this path, KFP automatically uploads the file to Minio Object Storage.
    - ("Numpy"): Simple metadata tag to show in the UI that this file is in Numpy format.
    """
    
    import tensorflow as tf
    import numpy as np

    print("Download data & preprocess...")
    mnist = tf.keras.datasets.mnist
    """
    Train set(x_train, y_train): Data to be used for model training.(60,000)
    Test set(x_test, y_test): Not involved in training, to be used only for final evaluation.(10,000)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # TF Keras's built-in dataset downloader.
    
    x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalization

    # Save Numpy array object in memory as .npy binary file format in disk.
    with open(x_train_path, 'wb') as f: np.save(f, x_train)
    with open(y_train_path, 'wb') as f: np.save(f, y_train)
    with open(x_test_path, 'wb') as f: np.save(f, x_test)
    with open(y_test_path, 'wb') as f: np.save(f, y_test)
    print("Complete preprocessing.")


@dsl.component(
    base_image='tensorflow/tensorflow:2.11.0', 
    packages_to_install=['numpy'] 
)
def train_model(
    x_train_path: dsl.InputPath("Numpy"),
    y_train_path: dsl.InputPath("Numpy"),
    epochs: int,
    model_file: dsl.OutputPath("Model")
):
    """
    Key steps to perform training and create a model.
    
    :param x_train_path: In the previous step(preprocess), the file uploaded to MinIO is automatically downloaded to the current container's local path. This variable contains the local file path.
    :type x_train_path: dsl.InputPath("Numpy")
    :param y_train_path: Same as x_train_path description.
    :type y_train_path: dsl.InputPath("Numpy")
    :param epochs: A hyperparameter that determines how many times the model will learn the entire dataset(60,000 items).
    :type epochs: int
    :param model_file: Same as x_train_path description
    :type model_file: dsl.OutputPath("Model")
    """
    
    import tensorflow as tf
    import numpy as np

    # Data load
    with open(x_train_path, 'rb') as f: x_train = np.load(f)
    with open(y_train_path, 'rb') as f: y_train = np.load(f)

    """
    Flatten:
    - Input: A two-dimensional array of 28 pixels wide and 28 pixels high (image). 
    - Action: This is flattened into a one-dimensional vector with 784 (2828) values. Becuase Dense layer can only receive one-dimensional data. 
    
    Dense: All 784 input values ​​and 128 neurons are fully connected. Internally, matrix mulitplication(Y=WX+b) occurs.
    - activation='relu': This function adds nonlinearity to the results of linear operations.
    
    Dropout: 
    - Action: During the learning process, 20% of the neurons are randomly disabled (set to 0).
    - Purpose: Increases model versatility by preventing overfitting (reliance on specific neurons)
    
    Dense(10): Ultimately, need to guess one of the number from 0 to 9, there are 10 output neurons.
    - activation='softmax': Convert 10 output values ​​to probability values ​​(total 1.0)  
    """
    # Deep learning models are built by stacking 'layers' one after another.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), 
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', # Optimization algorithm that finds the minimum value of a loss function.
                  loss='sparse_categorical_crossentropy', # Formula for calculating the error between the answer(integer) and the predicted value(probability).
                  metrics=['accuracy']) 

    model.fit(x_train, y_train, epochs=epochs)
    
    model.save(model_file) 
    print("Learning and Saving complete.")

@dsl.component(
    base_image='tensorflow/tensorflow:2.11.0',
    packages_to_install=['numpy']
)
def evaluate_model(
    x_test_path: dsl.InputPath("Numpy"),
    y_test_path: dsl.InputPath("Numpy"),
    model_file: dsl.InputPath("Model")
) -> float:
    """
    :params: model_file: The path to the model folder saved in MinIO in the previous step(train_model) downloaded to the current conatiner.
    """
    import tensorflow as tf
    import numpy as np

    with open(x_test_path, 'rb') as f: x_test = np.load(f)
    with open(y_test_path, 'rb') as f: y_test = np.load(f)
    
    # Deserialize model objects into memory from saved files
    model = tf.keras.models.load_model(model_file)

    loss, accuracy = model.evaluate(x_test, y_test) # Predictions are made by inputting test data into the learned model(forward propagation) and accuracy is calculated by comparing it with the correct answer.
    print(f"Final test accuracy: {accuracy}")
    
    return float(accuracy)

@dsl.pipeline(name='Fixed MNIST Pipeline')
def advanced_pipeline(epochs: int = 5):
    """
    task.outputs[...]
    - Indicates to the KFP comiler that the output of the task will be used as input for the next task.
    - K8s determines the order in which Pods should run first. 
    """
    preprocess_task = preprocess_data()
    
    train_task = train_model(
        x_train_path=preprocess_task.outputs['x_train_path'],
        y_train_path=preprocess_task.outputs['y_train_path'],
        epochs=epochs
    )
    
    evaluate_task = evaluate_model(
        x_test_path=preprocess_task.outputs['x_test_path'],
        y_test_path=preprocess_task.outputs['y_test_path'],
        model_file=train_task.outputs['model_file']
    )

if __name__ == '__main__':
    compiler.Compiler().compile(advanced_pipeline, 'mnist_fixed.yaml')
    print("Pipeline creation complete.")