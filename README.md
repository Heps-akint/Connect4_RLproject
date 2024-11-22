# **Connect Four AI: A Reinforcement Learning Journey**

*An advanced Connect Four AI powered by reinforcement learning, inspired by the history and evolution of AI.*

---

## **Table of Contents**

- [Introduction](#introduction)
- [Inspiration](#inspiration)
- [Original Prompt](#original-prompt)
- [Project Overview](#project-overview)
- [Reinforcement Learning and AI Concepts](#reinforcement-learning-and-ai-concepts)
  - [Historical Development](#historical-development)
  - [Implementation of Techniques](#implementation-of-techniques)
- [Advanced Techniques Inspired by AlphaGo and AlphaZero](#advanced-techniques-inspired-by-alphago-and-alphazero)
- [Implementation Details](#implementation-details)
  - [Technologies and Tools Used](#technologies-and-tools-used)
  - [Proficiencies Demonstrated](#proficiencies-demonstrated)
- [Usage](#usage)
  - [Training the AI](#training-the-ai)
  - [Playing Against the AI](#playing-against-the-ai)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [References](#references)

---

## **Introduction**

This project represents a comprehensive journey into building a sophisticated **Connect Four AI** using **reinforcement learning**. The AI is designed to learn and improve its gameplay autonomously, achieving exceptional performance against human opponents. Drawing inspiration from the historical development of AI and reinforcement learning, as well as advanced techniques from **AlphaGo** and **AlphaZero**, this project not only demonstrates technical proficiency but also serves as a stepping stone towards a career in AI research.

---

## **Inspiration**

The project was inspired by a series of insightful videos that chronicle the history and evolution of AI and reinforcement learning:

1. **[The Amazing History of Reinforcement Learning](https://www.youtube.com/watch?v=Dov68JsIC4g)**
2. **[ChatGPT: 30 Year History | How AI Learned to Talk](https://www.youtube.com/watch?v=OFS90-FX6pg)**
3. **[How AI Learned to Think](https://www.youtube.com/watch?v=PvDaPeQjxOE)**

These videos provided a deep understanding of how AI systems have evolved over time, particularly in learning, reasoning, and decision-making capabilities.

---

## **Original Prompt**

> **"Write the code to create, train, and play against an AI Connect 4 player. I will train the AI using my RTX 3060 Ti GPU, so keep that in mind. Use the knowledge from these videos on 'How AI Learned to Feel | History of Reinforcement Learning', 'ChatGPT: 30 Year History | How AI Learned to Talk', and 'How AI Learned to Think' in order to combine techniques the same way they did in the videos to create the best AI. I also want the code to be easily understandable and linked to the concepts talked about in the videos so I can follow along and understand how the topics covered in the video actually relate to building real-world AI systems. Make sure to explain the importance of each technique along with their corresponding relation to the videos and how it all works together."**

---

## **Project Overview**

### **Description**

This project involves creating a Connect Four AI player that can:

- Learn from scratch using reinforcement learning.
- Improve over time through self-play.
- Employ advanced AI techniques to optimize performance.
- Play against human opponents with a high level of proficiency.

### **Goals**

- **Implement Reinforcement Learning**: Utilize reinforcement learning algorithms to enable the AI to learn optimal strategies through trial and error.
- **Incorporate Advanced AI Techniques**: Apply methods inspired by AlphaGo and AlphaZero, such as Monte Carlo Tree Search (MCTS) and deep neural networks.
- **Understand AI Concepts**: Align the project's development with historical AI concepts and techniques discussed in the inspirational videos.
- **Demonstrate Proficiency**: Showcase skills relevant to AI research, including programming, machine learning, and problem-solving.

---

## **Reinforcement Learning and AI Concepts**

### **Historical Development**

The project's foundation is built upon the historical milestones in AI and reinforcement learning:

1. **Boxes and Beads (1960s)**:
   - **Donald Michie's Matchbox Educable Noughts and Crosses Engine (MENACE)**.
   - Demonstrated learning through reinforcement by adjusting physical beads in matchboxes representing game states.

2. **Samuel's Checkers Player (1959)**:
   - Utilized feature weights and self-play to improve performance.
   - Pioneered the concept of machines learning from experience without explicit programming.

3. **Temporal Difference Learning and TD-Gammon (1992)**:
   - **Gerald Tesauro's** work on using neural networks for game evaluation.
   - Introduced the idea of learning value functions to evaluate board states.

4. **Deep Learning and Neural Networks (2000s - 2010s)**:
   - The rise of deep neural networks capable of automatic feature discovery.
   - Enabled AI systems to handle more complex tasks without handcrafted features.

5. **AlphaGo and AlphaZero (2016 - 2017)**:
   - Combined MCTS with deep neural networks.
   - Achieved superhuman performance in Go through self-play and reinforcement learning.

### **Implementation of Techniques**

The project integrates these historical concepts:

- **Reinforcement Learning Algorithm**:
  - The AI learns by receiving rewards (wins) and penalties (losses).
  - Uses self-play to generate experience and improve strategies.

- **Neural Networks**:
  - Implements a deep neural network to approximate the policy and value functions.
  - Automatically learns features from the board states without manual feature engineering.

- **Monte Carlo Tree Search (MCTS)**:
  - Enhances decision-making by simulating possible future moves.
  - Balances exploration and exploitation to find optimal strategies.

- **Self-Play Mechanism**:
  - The AI plays against itself to continuously learn and adapt.
  - Mimics the approach used by AlphaGo Zero to learn without human data.

- **Temporal Difference Learning**:
  - Updates value estimates based on the difference between predicted and actual rewards.
  - Allows the AI to learn from incomplete sequences and improve predictions over time.

---

## **Advanced Techniques Inspired by AlphaGo and AlphaZero**

The project incorporates advanced techniques to enhance the AI's performance:

- **Policy and Value Networks**:
  - Separate networks to estimate the probability of selecting each move (policy) and the expected outcome (value).
  - Improves the AI's ability to evaluate board positions and choose optimal actions.

- **Residual Neural Networks**:
  - Utilizes residual connections to allow deeper networks without the vanishing gradient problem.
  - Enables the AI to learn more complex patterns and strategies.

- **Domain Randomization and Data Augmentation**:
  - Applies random transformations to training data to improve generalization.
  - Ensures the AI is robust against a variety of game scenarios.

- **GPU Acceleration with RTX 3060 Ti**:
  - Leverages the computational power of the GPU to train deep neural networks efficiently.
  - Allows for faster training iterations and the ability to handle larger models.

- **Learning Rate Scheduling and Gradient Clipping**:
  - Adjusts learning rates during training to optimize convergence.
  - Uses gradient clipping to prevent exploding gradients and stabilize training.

---

## **Implementation Details**

### **Technologies and Tools Used**

- **Programming Language**: Python
- **Libraries and Frameworks**:
  - **PyTorch**: For building and training neural networks.
  - **NumPy**: For numerical computations.
  - **Matplotlib**: For plotting training metrics.
  - **Jupyter Notebook**: For interactive development and visualization.
- **Hardware**:
  - **NVIDIA RTX 3060 Ti GPU**: Accelerates deep learning computations.
- **Version Control**: Git and GitHub for code management and collaboration.

### **Proficiencies Demonstrated**

- **Deep Learning and Neural Networks**:
  - Designing and implementing deep neural network architectures.
  - Understanding of residual networks and their benefits.

- **Reinforcement Learning**:
  - Applying RL algorithms to train an AI agent.
  - Implementing self-play mechanisms and reward systems.

- **Algorithm Optimization**:
  - Utilizing MCTS for efficient decision-making.
  - Employing advanced training techniques like learning rate scheduling.

- **Programming and Software Development**:
  - Writing clean, modular, and well-documented code.
  - Using version control systems effectively.

- **Data Analysis and Visualization**:
  - Monitoring training progress through metrics and visualizations.
  - Analyzing AI performance to identify areas for improvement.

- **Hardware Utilization**:
  - Leveraging GPU capabilities for accelerated training.
  - Managing computational resources efficiently.

---

## **Usage**

### **Training the AI**

To train the AI from scratch:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/connect-four-ai.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Training Script**:
   ```bash
   python train.py
   ```
   - The script will initiate self-play, training the neural network over multiple iterations.
   - Training parameters can be adjusted in the `config.py` file.

### **Playing Against the AI**

To play against the trained AI:

1. **Ensure the AI is Trained**:
   - Either complete the training process or download pre-trained weights.

2. **Run the Game Script**:
   ```bash
   python play.py
   ```
   - You'll be prompted to make moves against the AI.
   - The AI uses the trained model to make decisions in real-time.

---

## **Conclusion and Future Work**

This project demonstrates the application of advanced AI and reinforcement learning techniques to create a high-performing Connect Four AI. By aligning the development with historical AI concepts and implementing state-of-the-art methods inspired by AlphaGo and AlphaZero, the project showcases both technical proficiency and a deep understanding of AI principles.

**Future Enhancements**:

- **Expand to Other Games**: Apply the same framework to more complex games like chess or Go.
- **Enhance the Neural Network Architecture**: Experiment with different architectures, such as transformers.
- **Implement Distributed Training**: Utilize multiple GPUs or cloud resources to accelerate training.
- **Research Integration**: Explore the integration of the AI into research projects or publications.

---

## **References**

1. **The Amazing History of Reinforcement Learning**  
   *YouTube Video*: [https://www.youtube.com/watch?v=Dov68JsIC4g](https://www.youtube.com/watch?v=Dov68JsIC4g)

2. **ChatGPT: 30 Year History | How AI Learned to Talk**  
   *YouTube Video*: [https://www.youtube.com/watch?v=OFS90-FX6pg](https://www.youtube.com/watch?v=OFS90-FX6pg)

3. **How AI Learned to Think**  
   *YouTube Video*: [https://www.youtube.com/watch?v=PvDaPeQjxOE](https://www.youtube.com/watch?v=PvDaPeQjxOE)

4. **AlphaGo Zero: Learning from Scratch**  
   *DeepMind Blog*: [https://deepmind.com/blog/article/alphago-zero-starting-from-scratch](https://deepmind.com/blog/article/alphago-zero-starting-from-scratch)

5. **Reinforcement Learning: An Introduction** by Richard S. Sutton and Andrew G. Barto  
   *Book*: [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

6. **PyTorch Documentation**  
   *Website*: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
