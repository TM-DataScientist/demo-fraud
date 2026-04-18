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


"""時系列データの時間幅を保ちながらタイムスタンプを再配置する補助関数群。"""

# Helper functions to adjust the timestamps of our data
# while keeping the order of the selected events and
# the relative distance from one event to the other
# データ全体の並び順と各イベント間の相対距離を保ったまま、
# タイムスタンプだけを新しい期間へ写像する補助関数群。
import pandas as pd


def date_adjustment(
    sample: pd.Timestamp,
    data_max: pd.Timestamp,
    new_max: pd.Timestamp,
    old_data_period: pd.Timedelta,
    new_data_period: pd.Timedelta,
) -> pd.Timestamp:
    """
    Adjust a specific sample's date according to the original and new time periods
    元の期間と新しい期間の対応関係に合わせて、1件分の日時を再計算する。

    :param sample: The sample's timestamp
    :param data_max: The original data's max timestamp
    :param new_max: The new data's max timestamp
    :param old_data_period: The original data's time period
    :param new_data_period: The new data's time period
    :param sample: 対象サンプルのタイムスタンプ
    :param data_max: 元データにおける最大タイムスタンプ
    :param new_max: 新しい期間における最大タイムスタンプ
    :param old_data_period: 元データ全体の期間長
    :param new_data_period: 新しいデータ期間の長さ

    :returns: The adjusted timestamp
    :returns: 新しい期間へ線形に変換したタイムスタンプ
    """
    # 元データの末尾からどれだけ離れているかを比率で表し、
    # 新しい期間でも同じ比率になるように時間差を再計算する。
    sample_dates_scale = (data_max - sample) / old_data_period
    sample_delta = new_data_period * sample_dates_scale
    new_sample_ts = new_max - sample_delta
    return new_sample_ts


def adjust_data_timespan(
    dataframe: pd.DataFrame,
    timestamp_col: str = "timestamp",
    new_period: str = "2d",
    new_max_date_str: str = "now",
):
    """
    Adjust the dataframe timestamps to the new time period
    DataFrame 全体のタイムスタンプ列を、指定した新しい期間へ再配置する。

    :param dataframe: The dataframe to adjust
    :param timestamp_col: The timestamp column name
    :param new_period: The new time period
    :param new_max_date_str: The new max date
    :param dataframe: 変換対象の DataFrame
    :param timestamp_col: タイムスタンプ列名
    :param new_period: 再配置後の期間長
    :param new_max_date_str: 再配置後の末尾日時

    :returns: The adjusted dataframe
    :returns: タイムスタンプを再配置した DataFrame
    """
    # Calculate old time period
    # 元データがどれくらいの期間にまたがっているかを計算する。
    data_min = dataframe.timestamp.min()
    data_max = dataframe.timestamp.max()
    old_data_period = data_max - data_min

    # Set new time period
    # 指定された期間文字列と末尾日時から、新しい期間の境界を決める。
    new_time_period = pd.Timedelta(new_period)
    new_max = pd.Timestamp(new_max_date_str)
    new_min = new_max - new_time_period
    new_data_period = new_max - new_min

    # Apply the timestamp change
    # 破壊的変更を避けるためコピーを作成し、各行の日時を順番に変換する。
    df = dataframe.copy()
    df[timestamp_col] = df[timestamp_col].apply(
        lambda x: date_adjustment(
            x, data_max, new_max, old_data_period, new_data_period
        )
    )
    # 変換後はタイムスタンプ順に並べ直して、後続処理が扱いやすい形にする。
    df.sort_values(by="timestamp", axis=0, inplace=True)
    return df
