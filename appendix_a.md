---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Appendix A: Python/Jupyter Setup Deep Dive

+++

## Introduction

Welcome to the practical side of probability! This book relies heavily on using Python and specific libraries within the Jupyter Notebook environment to explore concepts, run simulations, and visualize results. This appendix provides a detailed guide to setting up this environment on your computer.

Our goal is to ensure you can:
1.  Install Python.
2.  Run Jupyter Notebooks.
3.  Install and verify the core scientific libraries: NumPy, SciPy, Matplotlib, and Seaborn.
4.  Access the book's code examples.

Even if you have some experience with Python, we recommend skimming through to ensure you have the specific setup used throughout the book.

+++

## 1. Python Installation: The Anaconda Distribution

The easiest way to get Python and the essential scientific libraries installed together is by using the **Anaconda Distribution**. Anaconda bundles Python, Jupyter, many core libraries (like NumPy, SciPy, Matplotlib, Pandas), and a package manager (`conda`) that simplifies installation and management of environments.

**Steps:**

1.  **Download:** Go to the Anaconda Distribution website: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
2.  **Select Installer:** Download the installer appropriate for your operating system (Windows, macOS, Linux). Choose the latest Python 3.x version.
3.  **Run Installer:** Execute the downloaded installer. Follow the on-screen instructions.
    * **Recommendation:** Unless you have a specific reason not to, accept the default settings. This usually includes adding Anaconda to your system's PATH environment variable (though the installer might advise against it for advanced users, for beginners it's often simpler).
    * You do **not** need administrator privileges to install Anaconda for your user only, which is often sufficient.

**Verification:**

Once installation is complete, open your system's terminal or command prompt:
* **Windows:** Search for `Anaconda Prompt` in the Start menu and open it.
* **macOS:** Open the `Terminal` application (found in Applications > Utilities).
* **Linux:** Open your standard terminal.

In the terminal, type the following command and press Enter:

```bash
python --version
```

You should see output similar to `Python 3.x.y` (e.g., `Python 3.10.9`), indicating Python is installed and accessible. If you get an error like "command not found", the installation might not have completed correctly, or Anaconda wasn't added to your PATH (you might need to reinstall or manually add it, consult Anaconda's documentation).

+++

## 2. Jupyter Notebooks

Jupyter Notebooks provide an interactive, browser-based environment where you can write and execute Python code, add explanatory text (like this!), include mathematical equations, and display visualizations, all in one document. Anaconda comes with Jupyter pre-installed.

**Launching JupyterLab (Recommended Interface):**

1.  Open the Anaconda Prompt (Windows) or Terminal (macOS/Linux).
2.  Navigate to the directory where you want to store your notebooks (or where you downloaded the book's code). You can use the `cd` (change directory) command. For example:
    ```bash
    # Example for Windows - navigate to a 'ProbBook' folder on the C drive
    cd C:\Users\YourUsername\Documents\ProbBook 
    
    # Example for macOS/Linux - navigate to a 'ProbBook' folder in your home directory
    cd ~/Documents/ProbBook 
    ```
3.  Type the following command and press Enter:
    ```bash
    jupyter lab
    ```
4.  This should automatically open a new tab in your default web browser displaying the JupyterLab interface. The terminal window must remain open while you are using JupyterLab.

**Alternatively, Launching Classic Jupyter Notebook:**

If you prefer the classic interface, use this command instead:
```bash
jupyter notebook
```

**Basic Jupyter Usage:**

* **Interface:** JupyterLab typically shows a file browser on the left and a main work area on the right. You can open existing `.ipynb` notebook files or create new ones (File > New > Notebook).
* **Cells:** Notebooks are composed of cells. The two main types are:
    * **Code Cells:** Contain Python code to be executed. Select a code cell and press `Shift + Enter` (or click the Run button) to execute the code. Output appears below the cell.
    * **Markdown Cells:** Contain formatted text (using Markdown syntax), like this cell. Press `Shift + Enter` to render the Markdown.
* **Kernel:** The 'engine' that executes the code. You can restart the kernel if things get stuck (Kernel > Restart Kernel...). 
* **Saving:** Notebooks are saved automatically, but you can manually save using File > Save Notebook.
* **Closing:** Close the browser tab. In the terminal where you launched Jupyter, press `Ctrl + C` (you might need to press it twice) and confirm shutdown if prompted.

+++

## 3. Core Libraries Installation/Verification

Anaconda typically installs the most crucial libraries for this book automatically. These include:

* **NumPy:** Fundamental package for numerical computing, especially arrays (`np`).
* **SciPy:** Library for scientific and technical computing, building on NumPy (`scipy`). We heavily use `scipy.stats`, `scipy.special`, and `scipy.integrate`.
* **Matplotlib:** Core plotting library (`plt`).
* **Pandas:** Library for data manipulation and analysis (often used alongside NumPy, `pd`). While not the primary focus, it's useful for handling datasets in examples.

**Seaborn**, a visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics (`sns`), might sometimes need to be installed separately.

**Checking and Installing Libraries:**

1.  **Using Conda (Recommended with Anaconda):**
    Open Anaconda Prompt or Terminal and use `conda install`:
    ```bash
    conda install numpy scipy matplotlib pandas seaborn
    ```
    Conda will check if the packages are installed and update them or install them if missing.

2.  **Using Pip (Alternative):**
    If you are not using Anaconda or prefer pip:
    ```bash
    pip install numpy scipy matplotlib pandas seaborn
    ```
    *Note: It's generally recommended to stick to one package manager (`conda` or `pip`) within a single environment to avoid potential conflicts.*

**Verification:**

The best way to verify is to create a new Jupyter Notebook and run the following code cell. If it executes without any `ImportError` messages, your core libraries are ready.

```{code-cell} ipython3
# Verification Cell: Import Core Libraries

import math
import platform # Moved import to the top
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import scipy.special
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Check versions (optional, but good for debugging)
print(f"Python version: {platform.python_version()}") 
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"SciPy version: {scipy.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")

# Basic functionality test
try:
    arr = np.array([1, 2, 3])
    mean_val = np.mean(arr)
    binom_prob = stats.binom.pmf(k=1, n=2, p=0.5)
    plt.figure()
    sns.histplot([1, 2, 2, 3, 3, 3])
    plt.close() # Prevents plot from displaying here
    print("\nCore libraries imported and basic functions tested successfully!")
except Exception as e:
    print(f"\nAn error occurred during testing: {e}")
    print("Please check your installations.")
```

## 4. Getting the Book's Code

All the Jupyter Notebooks containing the code examples and exercises for this book are available online, typically in a GitHub repository.

* **Location:** Please refer to the Preface or the book's companion website for the exact URL of the GitHub repository (e.g., `https://github.com/snowch/learn_probability`).

**How to Download:**

1.  **Download ZIP:** The simplest way is often to navigate to the main page of the GitHub repository in your web browser and look for a green "Code" button. Clicking it usually reveals a "Download ZIP" option. Download the file and unzip it into a convenient location on your computer (e.g., the `ProbBook` folder you might have created earlier).
2.  **Using Git (More Advanced):** If you are familiar with Git, you can clone the repository. Open your terminal or Anaconda Prompt, navigate to where you want to store the code, and run:
    ```bash
    git clone https://github.com/snowch/learn_probability.git
    ```
    This method makes it easier to get updates later using `git pull`.

Once downloaded, you can launch JupyterLab or Jupyter Notebook from the directory containing the code, and you should see the `.ipynb` files ready to be opened.

+++

## 5. Basic Troubleshooting

If you encounter issues, here are a few common problems and suggestions:

* **`command not found` (e.g., `python`, `jupyter`, `conda`):** This usually means Anaconda/Python is not in your system's PATH. Try using the `Anaconda Prompt` specifically, as it's pre-configured. If using a standard terminal, you might need to reinstall Anaconda (ensuring the 'Add to PATH' option is selected, if appropriate for your comfort level) or manually configure the PATH environment variable (consult Anaconda documentation).
* **`ImportError: No module named 'some_library'`:** The specific library (e.g., `seaborn`) is not installed in the Python environment Jupyter is using. Use `conda install some_library` or `pip install some_library` in the Anaconda Prompt/Terminal. Make sure you install it in the same environment Jupyter is running from (usually the 'base' environment if you haven't created others).
* **Code runs indefinitely or crashes:** Try restarting the Jupyter kernel (Kernel > Restart Kernel...). This often resolves issues caused by variables holding unexpected states.
* **Plots not showing:** Ensure you have run `%matplotlib inline` (usually included at the start of notebooks) or check for specific errors in the plotting code.
* **Permission Errors (conda/pip):** If you installed Anaconda system-wide, you might need administrator privileges to install new packages. Try running Anaconda Prompt 'as Administrator' (Windows) or using `sudo` (macOS/Linux, use with caution).
* **Searching for Help:** Copy and paste the error message into a search engine like Google. Include terms like `python`, `jupyter`, `conda`, and the library name. Stack Overflow ([stackoverflow.com](https://stackoverflow.com)) is an excellent resource for programming-related errors.

+++

## 6. Next Steps

With your environment set up and verified, you are ready to dive into the world of probability with Python! You can now proceed to **Chapter 1: Introduction to Probability and Python Setup** to begin your hands-on journey.

Make sure you can open and run the first few code cells in the Chapter 1 notebook provided with the book's materials. Happy coding and exploring!
