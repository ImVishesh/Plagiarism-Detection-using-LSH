# Plagiarism-Detection-using-LSH
🔍 **Plagiarism Detection using LSH in Python**

Developed an advanced plagiarism detection system using Python and NumPy, powered by the Locality Sensitive Hashing (LSH) algorithm. The project focused on analyzing the Auto & Property Insurance Dataset.

**Key Aspects**:

**LSH Algorithm Implementation**: Crafted the core LSH algorithm entirely from scratch in Python, showcasing strong coding skills.

**Numerical Processing with NumPy**: Utilized the NumPy library to handle numerical operations efficiently, enhancing the performance and scalability of the plagiarism detection system.

**Comparative Analysis**: Conducted an in-depth comparative analysis for different sets of hash functions (50, 100, and 200) to identify and report pairs with estimated similarity ≥ 0.5.

**Dataset Utilization**: Employed the Auto & Property Insurance Dataset to evaluate the effectiveness of the plagiarism detection system.

**LSH Optimization**: Fine-tuned the LSH implementation for plagiarism detection, optimizing parameters such as the number of hash functions and bands (r, b). Explored various (r, b) combinations, including (r=5, b=10) for 50 hash functions, (r=5, b=20) for 100 hash functions, (r=5, b=40) for 200 hash functions, and (r=10, b=20) for 200 hash functions.

**Threshold Analysis**: Analyzed the impact of modifying the similarity threshold to ≥ 0.8, investigating false positives and negative changes.

**Result Reporting**: Provided detailed reporting of false positives and negatives, averaged over five runs to ensure robustness and accuracy.
This project demonstrates proficiency in implementing a plagiarism detection system from scratch, optimizing LSH parameters, and analyzing its performance using real-world insurance data.
