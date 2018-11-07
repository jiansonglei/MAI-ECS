The code is for the paper:
@inproceedings{jian2019maie,
title={Evolutionarily Learning Multi-aspect Interactions and Influences from Network Structure and Node Content},
author={Jian, Songlei and Hu, Liang and Cao, Longbing and Lu, Kai and Gao, Hang},
year={2019},
organization={AAAI} }

1. Before running the code, please download the word embedding first. Please refers to load_attributed_network.py.
We use glove embedding and download at https://nlp.stanford.edu/projects/glove/.

2. If you use default data, you can directly run test_attributed_network.py.
   If you use new data, you need to prepare a docs.txt and adjedges.txt.
   Then you can generate caption.json and vocab.json from docs.txt through 'create_json_data' in load_attributed_network.py.

3. We evaluate the embedding through node classification and link prediction.
   Classificaiton: The node labels should be prepared.
   Link prediction: We construct the test file with 1 positive node and 20 negative node for one target node.
    And the ranking result is among the 20 negative nodes.

If you use the code, please cite the paper.