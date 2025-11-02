import unittest
from model import EmotionDetector  # Assuming EmotionDetector is the class for your model

class TestEmotionDetector(unittest.TestCase):

    def setUp(self):
        self.model = EmotionDetector()
        self.model.load_model('trained_models/emotion_detector_v1.h5')  # Load the trained model

    def test_model_prediction(self):
        # Test with a sample image
        sample_image = 'path/to/sample/image.jpg'  # Replace with an actual image path
        prediction = self.model.predict(sample_image)
        self.assertIn(prediction, ['happy', 'sad', 'angry', 'surprised', 'neutral'])  # Adjust based on your classes

    def test_model_training(self):
        # Test the training process
        training_data = 'path/to/training/data'  # Replace with actual training data path
        result = self.model.train(training_data)
        self.assertTrue(result)  # Assuming train method returns True on success

if __name__ == '__main__':
    unittest.main()