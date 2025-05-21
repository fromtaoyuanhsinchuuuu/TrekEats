# Project Title

## Overview

This project is a web application designed to help users extract information from menus. Users can upload images of menus, and the application will use Optical Character Recognition (OCR) to extract text. Additionally, the application provides features such as image search, nutrition information, reviews, and translation services for the menu items.

## Key Features

*   **OCR Text Extraction:** Extracts text from menu images.
*   **Image Search:** Allows users to search for images related to menu items.
*   **Nutrition Information:** Provides nutritional details for menu items.
*   **Reviews:** Displays user reviews and ratings for menu items.
*   **Translation:** Translates menu text into different languages.

## Project Structure

The project is divided into two main components: a frontend and a backend.

### Frontend

The frontend is a React application located in the `frontend` directory. It provides the user interface for interacting with the application.

*   **`frontend/src/App.js`**: The main application component.
*   **`frontend/src/components/`**: Contains reusable React components.
    *   **`frontend/src/components/MenuDetails.js`**: Displays the details of a menu.
    *   **`frontend/src/components/ReviewSummary.js`**: Summarizes reviews for menu items.
    *   **`frontend/src/components/UploadMenu.js`**: Handles menu image uploads.
*   **`frontend/src/pages/`**: Contains the main pages of the application.
    *   **`frontend/src/pages/Home.js`**: The landing page.
    *   **`frontend/src/pages/Results.js`**: Displays the results after processing a menu.
*   **`frontend/package.json`**: Lists the frontend dependencies and scripts.

### Backend

The backend is a Flask application located in the `backend` directory. It handles the business logic and API endpoints.

*   **`backend/app.py`**: The main Flask application file (Note: This file is currently empty and needs to be implemented).
*   **`backend/ocr.py`**: Contains the OCR logic using Tesseract and OpenCV.
*   **`backend/image_search.py`**: Handles image search functionality.
*   **`backend/nutrition.py`**: Provides nutrition information.
*   **`backend/reviews.py`**: Manages reviews and ratings.
*   **`backend/translate.py`**: Handles text translation.
*   **`backend/requirements.txt`**: Lists the backend Python dependencies.
*   **`backend/tests/`**: Contains backend tests.
    *   **`backend/tests/test_ocr.py`**: Tests for the OCR functionality.

## Backend Functionality

The backend provides several key functionalities through its API endpoints.

### OCR (Optical Character Recognition)

*   **Image Preprocessing:** Before text extraction, images undergo preprocessing steps to improve accuracy. This includes:
    *   Conversion to grayscale.
    *   Noise reduction using `cv2.fastNlMeansDenoising`.
    *   Adaptive thresholding using `cv2.adaptiveThreshold` (Gaussian method).
    *   Dilation to make text clearer.
*   **Text Extraction:** Uses `pytesseract` to extract text from the processed image. It supports both English and Korean languages (`-l eng+kor`).
*   **Text with Bounding Boxes:** Can extract text along with its coordinates (x, y, width, height) and confidence scores.
*   **Output:** Extracted text can be saved to a file and is returned in the API response.

### Other Backend Modules

*   **`image_search.py`**: Implements functionality to search for images related to menu items. (Details to be added based on implementation)
*   **`nutrition.py`**: Fetches and provides nutritional information for food items. (Details to be added based on implementation)
*   **`reviews.py`**: Manages user-submitted reviews and ratings for menu items. (Details to be added based on implementation)
*   **`translate.py`**: Provides translation services for menu text. (Details to be added based on implementation)

## Frontend Functionality

The frontend, built with React, provides an interactive user interface.

*   **Menu Upload:** Users can upload an image of a menu through the `UploadMenu` component.
*   **Results Display:** The `Results` page displays the information extracted from the menu, potentially including:
    *   The recognized text.
    *   Translated text.
    *   Nutrition information.
    *   Related images.
    *   User reviews.
*   **Menu Details:** The `MenuDetails` component is responsible for showing a structured view of the menu items and their associated information.
*   **Review Summaries:** The `ReviewSummary` component displays aggregated review scores or highlights.
*   **Navigation:** Users navigate between the `Home` page (for uploading) and the `Results` page.

## Setup and Running the Project

To get the project running locally, follow these steps:

### Prerequisites

*   Node.js and npm (for frontend)
*   Python 3 and pip (for backend)
*   Tesseract OCR engine (ensure it's installed and in your system's PATH)
    *   On macOS: `brew install tesseract tesseract-lang`
    *   On Ubuntu/Debian: `sudo apt-get update && sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-kor`
    *   Other systems: Refer to Tesseract documentation.

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Flask development server:**
    ```bash
    flask run
    ```
    (Note: `backend/app.py` needs to be implemented for this to work. Assuming it will be a standard Flask app.)

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    ```
3.  **Run the React development server:**
    ```bash
    npm start
    ```
    This will usually open the application in your default web browser at `http://localhost:3000`.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these general guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
    or
    ```bash
    git checkout -b fix/issue-number
    ```
3.  **Make your changes.**
    *   Ensure your code follows the project's coding style (if defined).
    *   Write clear and concise commit messages.
    *   Add or update tests as appropriate.
4.  **Test your changes thoroughly.**
5.  **Push your branch** to your fork:
    ```bash
    git push origin feature/your-feature-name
    ```
6.  **Create a Pull Request (PR)** against the main repository's `main` or `develop` branch.
    *   Provide a clear title and description for your PR, explaining the changes and why they are needed.
    *   Reference any related issues.

We appreciate your help in making this project better!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*This README was generated with assistance from an AI coding agent.*
