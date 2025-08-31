# Handwritten Digit Recognition using CNN and Custom Dataset


### Project Description: 

This project demonstrates how to build, train, and fine-tune a Convolutional Neural Network (CNN) to recognize handwritten digits. The model is initially trained on the standard MNIST dataset and then fine-tuned using a large, custom-collected dataset of handwritten digits to improve its real-world accuracy and robustness.

This repository is designed for beginners to understand the end-to-end process of a deep learning project, from data preparation to interactive testing.


## প্রজেক্টটি বাংলা বিবরণ:

এই প্রজেক্টটিতে দেখানো হয়েছে কীভাবে হাতে লেখা সংখ্যা শনাক্ত করার জন্য একটি কনভোলিউশনাল নিউরাল নেটওয়ার্ক (CNN) তৈরি, প্রশিক্ষণ এবং ফাইন-টিউন করা যায়। মডেলটিকে প্রথমে স্ট্যান্ডার্ড MNIST ডেটাসেটের উপর প্রশিক্ষণ দেওয়া হয় এবং তারপরে বাস্তব জগতের পারফরম্যান্স উন্নত করার জন্য হাতে লেখা সংখ্যার একটি বৃহৎ কাস্টম ডেটাসেট ব্যবহার করে ফাইন-টিউন করা হয়।



### What's This Project All About?

Have you ever wondered how your banking app can read the numbers from a check, or how the postal service sorts letters so efficiently? It’s all powered by something called Optical Character Recognition (OCR), and this project is my take on the foundational piece of that technology: recognizing handwritten digits.

I built a **Convolutional Neural Network (CNN)**—a special type of AI brain designed to "see" and understand images. My journey started with the famous **MNIST dataset**, which is like the "Hello, World!" for image recognition. But I didn't stop there. To make my model smarter and more robust, I decided to feed it a huge dataset of my own—**over 21,000 handwritten digits** that I collected!

This repository contains everything you need to follow along, from the datasets to the final, well-commented Google Colab notebook.



### এই প্রজেক্টটি আসলে কী?

কখনো ভেবে দেখেছেন, কীভাবে একটি ব্যাংকিং অ্যাপ চেকের উপর লেখা সংখ্যাগুলো পড়ে ফেলে, বা পোস্ট অফিস কীভাবে এত দ্রুত চিঠিগুলো সঠিক ঠিকানায় পাঠিয়ে দেয়? এর পেছনের প্রযুক্তিটি হলো অপটিক্যাল ক্যারেক্টার রিকগনিশন (OCR), আর এই প্রজেক্টটি হলো সেই প্রযুক্তির একেবারে মূল ভিত্তি—হাতে লেখা সংখ্যা চেনা।

এই কাজটি করার জন্য আমি একটি  কনভোলিউশনাল নিউরাল নেটওয়ার্ক (CNN)** তৈরি করেছি—এটি এক বিশেষ ধরনের কৃত্রিম বুদ্ধিমত্তা যা ছবি "দেখতে" এবং বুঝতে পারে। আমার এই যাত্রা শুরু হয়েছিল বিখ্যাত MNIST ডেটাসেট  দিয়ে, যা ছবি শনাক্তকরণের জগতে অনেকটা "অ-আ-ক-খ" শেখার মতো। কিন্তু আমি এখানেই থেমে থাকিনি। আমার মডেলটিকে আরও বাস্তবসম্মত এবং শক্তিশালী করার জন্য, আমি সেটিকে আমার নিজের সংগ্রহ করা ২১,০০০ এরও বেশি হাতে লেখা সংখ্যার একটি বিশাল ডেটাসেট দিয়ে প্রশিক্ষণ দিয়েছি।

এই রিপোজিটরিতে আপনি আমার পুরো কাজটি ধাপে ধাপে অনুসরণ করার জন্য প্রয়োজনীয় সবকিছুই পাবেন।

-

### 🚀 Features (বৈশিষ্ট্য)

- Model Architecture: A robust Convolutional Neural Network (CNN).
- Hybrid Training: Trained on a combination of standard MNIST data and a large custom dataset (~21,000+ images).
- High Accuracy: Achieves high accuracy on validation sets (~98-99%).
- Interactive Testing: Allows users to upload their own handwritten digit image and get a real-time prediction.
- Beginner-Friendly: The code is well-structured and organized in a Google Colab notebook for easy execution.

---

### Technology Used (ব্যবহৃত প্রযুক্তি)
- **Language:** Python
- **Libraries:**
  - TensorFlow & Keras
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Pillow (PIL)
- Environment: Google Colab



### Dataset (ডেটাসেট)
1. Standard MNIST Dataset: Provided by Kaggle (`train.csv`). Used for initial training.
2. Custom Handwritten Dataset: A large dataset of over 21,000 images (`Final_Custom_Dataset.zip`), collected and organized by digit (0-9) to fine-tune the model.



### How to Run (কীভাবে চালাবেন)

Follow these steps to run the project yourself:

**Step 1: Prepare Google Drive (গুগল ড্রাইভ প্রস্তুত করুন)**
- Download the `train.csv` and `Final_Custom_Dataset.zip` files from this repository.
- In your Google Drive, create a folder named `MNIST_CPP_Project`.
- Upload both downloaded files into this `MNIST_CPP_Project` folder.


**Step 2: Open in Google Colab (গুগল কোলাবে খুলুন)**
- Open the `Hand_Digit_Recognition.ipynb` file from this repository in Google Colab. (You can do this by navigating to Google Colab and selecting `File > Upload notebook`).


**Step 3: Run All Cells (সব সেল রান করুন)**
- In the Colab notebook, go to `Runtime > Run all`.
- The notebook will:
  1.  Connect to your Google Drive.
  2.  Load the standard MNIST dataset.
  3.  Unzip and load your custom dataset.
  4.  Combine both datasets.
  5.  Build and train the CNN model.
  6.  Finally, it will provide an upload button for you to test your own handwritten digit.

---

###  Project Structure (প্রজেক্টের গঠন)
```
├── Hand_Digit_Recognition.ipynb      # The main Google Colab notebook with all the code
├── Final_Custom_Dataset.zip          # The custom dataset with ~21,000 images
├── train.csv                         # The standard MNIST training data from Kaggle
└── README.md                         # This explanation file
```