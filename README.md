# DF-book-class-prediction

This is the code for CCF-BDCI Book Class Classification Task (Training Competition) 2023.

CCF-BDCI书籍文本分类问题（训练赛）代码仓库。

## Where do the functionalities hide? 如何查看这个代码？

As a data mining task, the code is a bit messy, but here is a guideline:
- All the code that is used for non language model fine tune models are exposed at the root directory.
- `train.py`, `test.py` are codes for GNNs. Depending on what models to run, you can specify it in the code.
- Codes with `_simple_classifier` are for text classifications that uses text embeddings. You can modify the classification method in the code.
- There is a dataloader wrapped in `utils.py`. The dataloader class is in `dataloader.py`. The dataloader loads the data in PyG dataset format, which could be used for both GNN methods and text classification methods. Also, the dataloader also contains capabilities for class balancing. The balancing code is implemented in `sampling/strategies.py`.
- All the models (potentially compatable with the main code structure) is in `models` folder.
- To generate sentenceBERT embeddings, the code is in `encoding` folder. Note that other embedding methods, the code is either in `other_approaces` or the `DataProcess&&MLapproaches` folder.
- Code for Fine-tune models is either in `other_approaces` or the `DataProcess&&MLapproaches` folder.
- You will have to download the source data and generate embedding yourself. If you would need the embedding files, please contact the owner of this repo.

作为一个数据挖掘任务，这段代码有点凌乱，但是这里有一些指南：
- 用于非语言模型微调模型的所有代码都暴露在根目录下。
- `train.py`、`test.py`是用于GNN的代码。根据要运行的模型，你可以在代码中指定。
- 带有`_simple_classifier`的代码用于使用文本嵌入的文本分类。你可以在代码中修改分类方法。
- `utils.py`中有一个数据加载器。数据加载器类位于`dataloader.py`中。数据加载器以PyG数据集格式加载数据，可用于GNN方法和文本分类方法。此外，数据加载器还包含类平衡的功能。平衡代码实现在`sampling/strategies.py`中。
- 所有模型（与主要代码结构兼容的可能）都在models文件夹中。
- 要生成sentenceBERT嵌入，代码位于encoding文件夹中。请注意，其他嵌入方法的代码要么在`other_approaces`文件夹中，要么在`DataProcess&&MLapproaches`文件夹中。
- 精调模型要么在`other_approaces`文件夹中，要么在`DataProcess&&MLapproaches`文件夹中。
- 你将需要下载源数据并自行生成嵌入。如果你需要嵌入文件，请联系此仓库的所有者。