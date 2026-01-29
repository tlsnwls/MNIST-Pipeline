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
    model_file: dsl.OutputPath("Model") # 확장자 없는 경로
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

    # 데이터 로드
    with open(x_train_path, 'rb') as f: x_train = np.load(f)
    with open(y_train_path, 'rb') as f: y_train = np.load(f)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # 
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs)
    
    model.save(model_file) 
    print("학습 및 저장 완료!")

@dsl.component(
    base_image='tensorflow/tensorflow:2.11.0',
    packages_to_install=['numpy']
)
def evaluate_model(
    x_test_path: dsl.InputPath("Numpy"),
    y_test_path: dsl.InputPath("Numpy"),
    model_file: dsl.InputPath("Model")
) -> float:
    import tensorflow as tf
    import numpy as np

    with open(x_test_path, 'rb') as f: x_test = np.load(f)
    with open(y_test_path, 'rb') as f: y_test = np.load(f)
    
    # 저장된 모델 불러오기 (폴더 형식 자동 인식)
    model = tf.keras.models.load_model(model_file)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"최종 테스트 정확도: {accuracy}, 손실률: {loss}")
    
    return float(accuracy)

# ---------------------------------------------------------
# 파이프라인 연결
# ---------------------------------------------------------
@dsl.pipeline(name='Fixed MNIST Pipeline')
def advanced_pipeline(epochs: int = 5):
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
    print("파이프라인 생성 완료.")