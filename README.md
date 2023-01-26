# Model API
Esta aplicación fue desarrollada en Python y haciendo uso del marco de trabajo web FastAPI. Esta API cuenta con todos los modelos entrenados en el desarrollo de esta investigación y se encarga de cargar solamente los necesarios para realizar la clasificación y predicción correspondiente. 

Esta aplicación expone un servicio llamado predict. El cual se encarga de tomar la información que es enviada desde la aplicación Backend y enviarla a los diferentes modelos entrenados (1) DTC, (2) RFC, (3) LRC, (4) LSTM y (5) GRU. Es necesario recordar que para los modelos de clasificación primero se llama al modelo especializado en detectar estados críticos, y si el resultado es “inestable”, se llama al modelo especializado en detectar estados estables. En total, esta aplicación cuenta con 32 modelos entrenados para calcular el resultado de la clasificación y la predicción de los datos.

Finalmente, luego de que los resultados son calculados se Model API envía una respuesta al Backend con esta información. Por su parte, el Backend se encargará de enviar la respuesta al Frontend, que a su vez, representará la información al usuario.
