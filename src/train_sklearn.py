# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""不正取引検知モデルの学習データ準備と学習処理。"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def prepare_data_to_train(
    transactions_data_p: pd.DataFrame,
    user_events_data_p: pd.DataFrame,
    labels_set: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function prepare data to train and test
    取引データ・ユーザーイベント・ラベルを結合し、学習用と評価用の分割を作成する。

    :param transactions_data_p: transactions data
    :param user_events_data_p: user events data
    :param labels_set: labels data
    :param transactions_data_p: 取引データ
    :param user_events_data_p: ユーザーイベントデータ
    :param labels_set: 正解ラベルデータ
    :return: train and test data
    :return: train_test_split が返す学習用・評価用データ一式
    """
    # モデル学習に使わない列を削除し、時系列結合できるよう時刻順に整列する。
    transactions_data_p.drop(columns=["age", "target", "device"], inplace=True)
    transactions_data_p.sort_values(by="timestamp", inplace=True)
    user_events_data_p.sort_values(by="timestamp", inplace=True)

    # 同一 source ごとに、最も近い過去のイベントを transaction に対応付ける。
    merged_df = pd.merge_asof(
        transactions_data_p,
        user_events_data_p,
        on="timestamp",
        by="source",
    )

    # ラベルも時系列に沿って結合し、学習に不要な識別子列を外した上で欠損行を除く。
    data_for_train = (
        pd.merge_asof(merged_df, labels_set, on="timestamp", by="source")
        .drop(columns=["source", "timestamp"])
        .dropna()
    )

    # 正解ラベルだけを別に保持し、説明変数からは取り除く。
    lable = data_for_train["label"]
    data_for_train.drop(columns=["label"], inplace=True)

    # 再現性を持たせるため random_state を固定して訓練/テストへ分割する。
    return train_test_split(data_for_train, lable, test_size=0.2, random_state=42)


def train_and_val(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> RandomForestClassifier:
    """
    This function train and validate the model
    ランダムサーチで RandomForest のハイパーパラメータを探索し、評価指標を表示する。

    :param X_train: train data
    :param X_test: test data
    :param y_train: train labels
    :param y_test: test labels
    :param X_train: 学習用特徴量
    :param X_test: 評価用特徴量
    :param y_train: 学習用ラベル
    :param y_test: 評価用ラベル
    :return: model
    :return: 最良パラメータで学習済みの RandomForestClassifier
    """
    # 探索対象の候補を広めに定義し、ランダムサーチで現実的な計算時間に抑える。
    grid_search = {
        "bootstrap": [True, False],
        "max_depth": [
            10,
            30,
            50,
            100,
        ],
        "max_features": ["log2", "sqrt"],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "n_estimators": [50, 100, 500],
    }

    # 乱数シードを固定した RandomizedSearchCV で最良モデルを探索する。
    rf = RandomForestClassifier()
    rfc = RandomizedSearchCV(
        estimator=rf,
        param_distributions=grid_search,
        n_iter=100,
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=-1,
    )
    rfc.fit(X_train, y_train)

    # Make predictions on the test set
    # テストデータに対する予測値を作成する。
    y_pred = rfc.best_estimator_.predict(X_test)

    # Calculate evaluation metrics
    # 分類性能を複数の観点で確認し、モデル比較しやすいように数値化する。
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the evaluation metrics
    # 学習ジョブのログからすぐ確認できるよう、主要メトリクスを出力する。
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    return rfc.best_estimator_
