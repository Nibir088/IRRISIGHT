import numpy as np
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

class Metric(ABC):
    """Abstract base class for metrics."""
    
    def __init__(self, num_classes: int, epsilon: float = 1e-7):
        self.num_classes = num_classes
        self.epsilon = epsilon
        # Remove reset() call from here - let subclasses handle their own initialization
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric states."""
        pass
    
    @abstractmethod
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update metric states with new predictions."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute the metric value."""
        pass

class ConfusionMatrix(Metric):
    """Confusion Matrix metric."""
    
    def __init__(self, num_classes: int, epsilon: float = 1e-7):
        super().__init__(num_classes, epsilon)
        self.reset()
    
    def reset(self) -> None:
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        mask = (y_true >= 0) & (y_true < self.num_classes)
        confusion_matrix = np.bincount(
            self.num_classes * y_true[mask].astype(int) + y_pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion_matrix
    
    def compute(self) -> np.ndarray:
        # print(np.sum(self.confusion_matrix, axis=0), np.sum(self.confusion_matrix, axis=1))
        return self.confusion_matrix

class Precision(Metric):
    """Precision metric."""
    
    def __init__(self, num_classes: int, average: str = 'micro', epsilon: float = 1e-7):
        super().__init__(num_classes, epsilon)
        self.average = average
        self.conf_matrix = ConfusionMatrix(num_classes)
        self.reset()  # Call reset after initializing conf_matrix
    
    def reset(self) -> None:
        self.conf_matrix.reset()
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.conf_matrix.update(y_true, y_pred)
    
    def compute(self) -> Union[float, np.ndarray]:
        conf_mat = self.conf_matrix.compute()
        tp = np.diag(conf_mat)
        # print()
        fp = np.sum(conf_mat, axis=0) - tp
        denominator = tp + fp

        # Handle case where denominator is zero
        precision = np.where(denominator > 0, tp / (denominator + self.epsilon), np.nan)

        if self.average == 'micro':
            return np.nansum(tp) / (np.nansum(tp) + np.nansum(fp) + self.epsilon)
        elif self.average == 'macro':
            return np.nanmean(precision)
        elif self.average == 'macro_irr':
            return np.nanmean(precision[1:])
        else:  # per-class
            return precision


class Recall(Metric):
    """Recall metric."""
    
    def __init__(self, num_classes: int, average: str = 'micro', epsilon: float = 1e-7):
        super().__init__(num_classes, epsilon)
        self.average = average
        self.conf_matrix = ConfusionMatrix(num_classes)
        self.reset()  # Call reset after initializing conf_matrix
    
    def reset(self) -> None:
        self.conf_matrix.reset()
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.conf_matrix.update(y_true, y_pred)
    
    def compute(self) -> Union[float, np.ndarray]:
        conf_mat = self.conf_matrix.compute()
        
        
        tp = np.diag(conf_mat)
        fn = np.sum(conf_mat, axis=1) - tp
        denominator = tp + fn

        # Handle case where denominator is zero
        recall = np.where(denominator > 0, tp / (denominator + self.epsilon), np.nan)

        if self.average == 'micro':
            return np.nansum(tp) / (np.nansum(tp) + np.nansum(fn) + self.epsilon)
        elif self.average == 'macro':
            return np.nanmean(recall)
        elif self.average == 'macro_irr':
            return np.nanmean(recall[1:])
        else:  # per-class
            return recall

class F1Score(Metric):
    """F1 Score metric."""
    
    def __init__(self, num_classes: int, average: str = 'micro', epsilon: float = 1e-7):
        super().__init__(num_classes, epsilon)
        self.average = average
        self.conf_matrix = ConfusionMatrix(num_classes)
        self.reset()  # Call reset after initializing conf_matrix
    
    def reset(self) -> None:
        self.conf_matrix.reset()
        # self.recall.reset()
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.conf_matrix.update(y_true, y_pred)
    
    def compute(self) -> Union[float, np.ndarray]:
        conf_mat = self.conf_matrix.compute()
        
        
        tp = np.diag(conf_mat)
        fn = np.sum(conf_mat, axis=1) - tp
        fp = np.sum(conf_mat, axis=0) - tp
        
        denominator = 2 * tp + fn + fp
        
        
#         precision = self.precision.compute()
#         recall = self.recall.compute()

#         denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * tp / denominator, np.nan)
        if self.average == 'micro':
            micro_tp = np.sum(tp)
            micro_fn = np.sum(fn)
            micro_fp = np.sum(fp)

            denominator = 2 * micro_tp + micro_fn + micro_fp

            # Handle division by zero correctly
            micro_f1 = np.where(denominator > 0, (2 * micro_tp) / (denominator + self.epsilon), 0.0)

            return micro_f1
        elif self.average == 'macro':
            return np.nanmean(f1)
        elif self.average == 'macro_irr':
            return np.nanmean(f1[1:])
        else:  # per-class
            return f1
        

class IoU(Metric):
    """IoU (Intersection over Union) metric."""
    
    def __init__(self, num_classes: int, average: str = 'micro', epsilon: float = 1e-7):
        super().__init__(num_classes, epsilon)
        self.average = average
        self.conf_matrix = ConfusionMatrix(num_classes)
        self.reset()  # Call reset after initializing conf_matrix
    
    def reset(self) -> None:
        self.conf_matrix.reset()
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.conf_matrix.update(y_true, y_pred)
    
    def compute(self) -> Union[float, np.ndarray]:
        conf_mat = self.conf_matrix.compute()
        tp = np.diag(conf_mat)
        fp = np.sum(conf_mat, axis=0) - tp
        fn = np.sum(conf_mat, axis=1) - tp
        denominator = tp + fp + fn

        # Handle case where denominator is zero
        iou = np.where(denominator > 0, tp / (denominator + self.epsilon), np.nan)

        if self.average == 'micro':
            return np.nansum(tp) / (np.nansum(tp) + np.nansum(fp) + np.nansum(fn) + self.epsilon)
        elif self.average == 'macro':
            return np.nanmean(iou)
        elif self.average == 'macro_irr':
            return np.nanmean(iou[1:])
        else:  # per-class
            return iou

class SegmentationMetrics:
    """Class to handle all segmentation metrics."""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.metrics = {
            'precision': {
                'micro': Precision(num_classes, 'micro'),
                'macro': Precision(num_classes, 'macro'),
                'macro_irr': Precision(num_classes, 'macro_irr'),
                'per_class': Precision(num_classes, 'none')
            },
            'recall': {
                'micro': Recall(num_classes, 'micro'),
                'macro': Recall(num_classes, 'macro'),
                'per_class': Recall(num_classes, 'none'),
                'macro_irr': Recall(num_classes, 'macro_irr')
            },
            'f1': {
                'micro': F1Score(num_classes, 'micro'),
                'macro': F1Score(num_classes, 'macro'),
                'per_class': F1Score(num_classes, 'none'),
                'macro_irr': F1Score(num_classes, 'macro_irr')
            },
            'iou': {
                'micro': IoU(num_classes, 'micro'),
                'macro': IoU(num_classes, 'macro'),
                'per_class': IoU(num_classes, 'none'),
                'macro_irr': IoU(num_classes, 'macro_irr')
            }
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        for metric_group in self.metrics.values():
            for metric in metric_group.values():
                metric.reset()
    
    def update(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> None:
        """
        Update all metrics with new predictions.
        
        Args:
            y_true: Ground truth masks of shape (H, W) or (B, H, W)
            y_pred_probs: Predicted probabilities of shape (num_classes, H, W) or (B, num_classes, H, W)
        """
        # print(y_pred_probs.shape)
        # Handle batch dimension
        if len(y_pred_probs.shape) == 4:
            y_pred = np.argmax(y_pred_probs, axis=1)
            for i in range(y_pred.shape[0]):
                self._update_single(y_true[i], y_pred[i])
        else:
            y_pred = np.argmax(y_pred_probs, axis=0)
            self._update_single(y_true, y_pred)
    
    def _update_single(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update metrics with a single prediction."""
        for metric_group in self.metrics.values():
            for metric in metric_group.values():
                metric.update(y_true, y_pred)
    
    def compute(self) -> Dict:
        """Compute all metrics and handle NaN values gracefully."""
        results = {}
        for metric_name, metric_group in self.metrics.items():
            results[metric_name] = {
                'micro': np.nan_to_num(metric_group['micro'].compute(), nan=0.0),
                'macro': np.nan_to_num(metric_group['macro'].compute(), nan=0.0),
                'macro_irr': np.nan_to_num(metric_group['macro_irr'].compute(), nan=0.0),
                'per_class': metric_group['per_class'].compute()
            }
        return results



# class ConfusionMatrix(Metric):
#     """Confusion Matrix metric."""
    
#     def __init__(self, num_classes: int, epsilon: float = 1e-7):
#         super().__init__(num_classes, epsilon)
#         self.reset()
    
#     def reset(self) -> None:
#         self.confusion_matrices = []  # Store per-sample confusion matrices
    
#     def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
#         """Compute confusion matrix per sample and store it."""
#         mask = (y_true >= 0) & (y_true < self.num_classes)
#         confusion_matrix = np.bincount(
#             self.num_classes * y_true[mask].astype(int) + y_pred[mask],
#             minlength=self.num_classes ** 2
#         ).reshape(self.num_classes, self.num_classes)
        
#         self.confusion_matrices.append(confusion_matrix)  # Store per-sample matrix
    
#     def compute(self, method: str = 'weighted') -> Union[np.ndarray, List[np.ndarray]]:
#         """Compute the aggregated confusion matrix."""
#         if method == 'weighted':
#             return np.sum(self.confusion_matrices, axis=0)
#         elif method == 'unweighted':
#             return self.confusion_matrices  # Return per-sample matrices
#         else:
#             raise ValueError("Invalid method. Choose 'weighted' or 'unweighted'.")
# class Precision(Metric):
#     """Precision metric."""
    
#     def __init__(self, num_classes: int, average: str = 'micro', epsilon: float = 1e-7):
#         super().__init__(num_classes, epsilon)
#         self.average = average
#         self.conf_matrix = ConfusionMatrix(num_classes)
#         self.reset()  # Call reset after initializing conf_matrix
    
#     def reset(self) -> None:
#         self.conf_matrix.reset()
    
#     def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
#         self.conf_matrix.update(y_true, y_pred)
    
#     def compute(self, method: str = 'weighted') -> Union[float, np.ndarray]:
        
#         conf_mat = self.conf_matrix.compute(method)
        
#         if method == 'weighted':
#             tp = np.diag(conf_mat)
#             # print()
#             fp = np.sum(conf_mat, axis=0) - tp
#             denominator = tp + fp

#             # Handle case where denominator is zero
#             precision = np.where(denominator > 0, tp / (denominator + self.epsilon), np.nan)

#             if self.average == 'micro':
#                 return np.nansum(tp) / (np.nansum(tp) + np.nansum(fp) + self.epsilon)
#             elif self.average == 'macro':
#                 return np.nanmean(precision)
#             elif self.average == 'macro_irr':
#                 return np.nanmean(precision[1:])
#             else:  # per-class
#                 return precision
#         else:
            
