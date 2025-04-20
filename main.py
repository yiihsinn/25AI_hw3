import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger
from sklearn.metrics import accuracy_score

from decision_tree import ConvNet, DecisionTree, get_features_and_labels, get_features_and_paths
from CNN import CNN, train, validate, test
from utils import TrainDataset, TestDataset, load_train_dataset, load_test_dataset, plot

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Start loading data")
    images, labels = load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)
    train_len = int(0.8 * len(images))
    train_images, val_images = images[:train_len], images[train_len:]
    train_labels, val_labels = labels[:train_len], labels[train_len:]
    test_images = load_test_dataset()

    train_dataset = TrainDataset(train_images, train_labels)
    val_dataset = TrainDataset(val_images, val_labels)
    test_dataset = TestDataset(test_images)

    logger.info("Start training CNN")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []
    max_acc = 0
    best_model = None

    EPOCHS = 10
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > max_acc:
            max_acc = val_acc
            best_model = model.state_dict()
            torch.save(best_model, 'best_model.pth')

    logger.info(f"Best Accuracy: {max_acc:.4f}")

    model.load_state_dict(torch.load('best_model.pth'))
    plot(train_losses, val_losses)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test(model, test_loader, device)

    logger.info("Start training Decision Tree")
    conv_model = ConvNet().to(device)
    conv_model.eval()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_features, train_labels = get_features_and_labels(conv_model, train_loader, device)
    train_features_df = pd.DataFrame(train_features)
    tree_model = DecisionTree(max_depth=7)
    tree_model.fit(train_features_df, train_labels)

    val_features, val_labels = get_features_and_labels(conv_model, val_loader, device)
    val_features_df = pd.DataFrame(val_features)
    val_predictions = tree_model.predict(val_features_df)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_features, test_paths = get_features_and_paths(conv_model, test_loader, device)
    test_features_df = pd.DataFrame(test_features)
    test_predictions = tree_model.predict(test_features_df)

    results = pd.DataFrame({'id': test_paths, 'prediction': test_predictions})
    results.to_csv('DecisionTree.csv', index=False)
    print("Predictions saved to 'DecisionTree.csv'")

if __name__ == '__main__':
    main()