# Decentralized Multi-Agent Contrastive Representation Learning (D-MACRL)

## Abstract
In an era dominated by large single-agent models, collective intelligence through decentralized multi-agent systems presents a promising paradigm for enhancing representation learning while addressing data privacy concerns and mitigating risks of single points of failure. This project introduces the D-MACRL model, which utilizes state-of-the-art contrastive learning techniques like SimCLR to foster collaboration among intelligent agents without the need for a fusion center. The primary goal is to develop a system where agents independently learn and share representations, based on locally unique data, to improve each other's local models through collaboration via a graph-based network. A significant component of this research involves the integration of the Exact Diffusion algorithm to facilitate effective learning even within sparsely connected agent networks. The project explores how the connectivity of the agent networks impacts the efficiency and effectiveness of decentralized learning systems. Through a comparative study with centralized models like SimCLR and a federated approach, this project evaluates the performance of D-MACRL in image classification tasks using both IID and non-IID data distributions. The results aim to demonstrate that local shared representations can achieve performance comparable to larger, centralized models and analyze the effectiveness of exact diffusion compared to regular diffusion in decentralized settings when the connections between agents are sparse.

## User Guide

This guide provides a straightforward procedure for setting up and running the model. The steps outlined below are designed to ensure ease of use for individuals at all levels of technical expertise.

### Getting Started
The code and the data gathered for the D-MACRL model can be accessed from the following GitHub repository:
[https://github.com/kert114/D-MACRL.git](https://github.com/kert114/D-MACRL.git).

### Setup and Execution
1. It is advisable to run the model on a remote server rather than a local computer to reduce run times significantly.
2. Connect to the desired machine using SSH.
3. Clone the project repository from GitHub to your machine.
4. Execute the `bash require.sh` command in the terminal to install all required libraries and set up a virtual environment automatically.
5. Create or edit a bash script to configure the model settings as per your requirements. You can find detailed explanations of all configurable options in the `options.py` file.
6. Start the model by running the bash script `bash run_model.sh` with the appropriate arguments.
7. The model’s output will be automatically saved to a text file, and TensorBoard event files will be stored in the `/save` folder.
8. To view the TensorBoard logs, execute `tensorboard --logdir="/path to folder"/fed_sim/save --host localhost --port 8087`.
9. Open a web browser and navigate to `http://localhost:8087` to access the TensorBoard interface and view detailed metrics of the model's performance.

