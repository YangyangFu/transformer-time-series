# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Directory to extract time covariates.

Extract time covariates from datetime.
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import EasterMonday
from pandas.tseries.holiday import GoodFriday
from pandas.tseries.holiday import Holiday
from pandas.tseries.holiday import SU
from pandas.tseries.holiday import TH
from pandas.tseries.holiday import USColumbusDay
from pandas.tseries.holiday import USLaborDay
from pandas.tseries.holiday import USMartinLutherKingJr
from pandas.tseries.holiday import USMemorialDay
from pandas.tseries.holiday import USPresidentsDay
from pandas.tseries.holiday import USThanksgivingDay
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import Day
from pandas.tseries.offsets import Easter
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# This is 183 to cover half a year (in both directions), also for leap years
# + 17 as Eastern can be between March, 22 - April, 25
MAX_WINDOW = 183 + 17


def _distance_to_holiday(holiday):
  """Return distance to given holiday."""

  def _distance_to_day(index):
    holiday_date = holiday.dates(
        index - pd.Timedelta(days=MAX_WINDOW),
        index + pd.Timedelta(days=MAX_WINDOW),
    )
    assert (
        len(holiday_date) != 0  # pylint: disable=g-explicit-length-test
    ), f"No closest holiday for the date index {index} found."
    # It sometimes returns two dates if it is exactly half a year after the
    # holiday. In this case, the smaller distance (182 days) is returned.
    return (index - holiday_date[0]).days

  return _distance_to_day

EasterSunday = Holiday(
    "Easter Sunday", month=1, day=1, offset=[Easter(), Day(0)]
)
NewYearsDay = Holiday("New Years Day", month=1, day=1)
SuperBowl = Holiday(
    "Superbowl", month=2, day=1, offset=DateOffset(weekday=SU(1))
)
MothersDay = Holiday(
    "Mothers Day", month=5, day=1, offset=DateOffset(weekday=SU(2))
)
IndependenceDay = Holiday("Independence Day", month=7, day=4)
ChristmasEve = Holiday("Christmas", month=12, day=24)
ChristmasDay = Holiday("Christmas", month=12, day=25)
NewYearsEve = Holiday("New Years Eve", month=12, day=31)
BlackFriday = Holiday(
    "Black Friday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=TH(4)), Day(1)],
)
CyberMonday = Holiday(
    "Cyber Monday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=TH(4)), Day(4)],
)

HOLIDAYS = [
    EasterMonday,
    GoodFriday,
    USColumbusDay,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    EasterSunday,
    NewYearsDay,
    SuperBowl,
    MothersDay,
    IndependenceDay,
    ChristmasEve,
    ChristmasDay,
    NewYearsEve,
    BlackFriday,
    CyberMonday,
]

"""
def _which_holiday(holiday):
  def _holiday_(index):
    holiday_idx = HOLIDAYS.index(holiday)
    
    # if the holiday is observed within given window, return the holiday index
    dates = holiday.dates(
        index.date() - pd.Timedelta(days=0),
        index.date() + pd.Timedelta(days=0),
    )
    if len(dates) > 0:
      return holiday_idx+1
      
    # if no holiday found, return 0
    return 0
  
  return _holiday_
"""  

def _which_holiday(index):
    """Return the index of the holiday if it is the holiday."""
    for holiday_idx, holiday in enumerate(HOLIDAYS):
      # if the holiday is observed within given window, return the holiday index
      dates = holiday.dates(
          index.date() - pd.Timedelta(days=0),
          index.date() + pd.Timedelta(days=0),
      )
      if len(dates) > 0:
        return holiday_idx+1
      
    # if no holiday found, return 0
    return 0  

class TimeCovariates(object):
  """Extract all time covariates except for holidays."""

  def __init__(
      self,
      datetimes,
      normalized = False,
      use_holiday_distance = False,
      use_which_holiday = False,
  ):
    """Init function.

    Args:
      datetimes: pandas DatetimeIndex (lowest granularity supported is min)
      normalized: whether to normalize features or not
      use_holiday_distance: whether to use holiday distance features or not, this will construct 18 features for each time stamp
      use_which_holiday: whether to use which holiday features or not, this will construct 1 feature for each time stamp, indicating which holiday the timestamp is in

    Returns:
      None
    """
    self.normalized = normalized
    self.dti = datetimes
    self.use_holiday_distance = use_holiday_distance
    self.use_which_holiday = use_which_holiday
    
  def _minute_of_hour(self):
    minutes = np.array(self.dti.minute, dtype=np.float32)
    if self.normalized:
      minutes = minutes / 59.0 - 0.5
    return minutes

  def _hour_of_day(self):
    hours = np.array(self.dti.hour, dtype=np.float32)
    if self.normalized:
      hours = hours / 23.0 - 0.5
    return hours

  def _day_of_week(self):
    day_week = np.array(self.dti.dayofweek, dtype=np.float32)
    if self.normalized:
      day_week = day_week / 6.0 - 0.5
    return day_week

  def _day_of_month(self):
    day_month = np.array(self.dti.day, dtype=np.float32)
    if self.normalized:
      day_month = day_month / 30.0 - 0.5
    return day_month

  def _day_of_year(self):
    day_year = np.array(self.dti.dayofyear, dtype=np.float32)
    if self.normalized:
      day_year = day_year / 364.0 - 0.5
    return day_year

  def _month_of_year(self):
    month_year = np.array(self.dti.month, dtype=np.float32)
    if self.normalized:
      month_year = month_year / 11.0 - 0.5
    return month_year

  def _week_of_year(self):
    week_year = np.array(self.dti.strftime("%U").astype(int), dtype=np.float32)
    if self.normalized:
      week_year = week_year / 51.0 - 0.5
    return week_year

  def _get_distance_holidays(self):
    dti_series = self.dti.to_series()
    hol_variates = np.vstack(
        [
            dti_series.apply(_distance_to_holiday(h)).values
            for h in tqdm(HOLIDAYS)
        ]
    )
    # hol_variates is (num_holiday, num_time_steps), the normalization should be
    # performed in the num_time_steps dimension.
    return StandardScaler().fit_transform(hol_variates.T).T

  def _get_which_holiday(self):
    dti_series = self.dti.to_series()
    hol_variates = np.hstack([_which_holiday(t) for t in tqdm(dti_series)])

    # hol_variates is (num_holiday, num_time_steps), the normalization should be
    # performed in the num_time_steps dimension.
    return hol_variates

  def get_covariates(self):
    """Get all time covariates."""
    moh = self._minute_of_hour().reshape(1, -1)
    hod = self._hour_of_day().reshape(1, -1)
    dom = self._day_of_month().reshape(1, -1)
    dow = self._day_of_week().reshape(1, -1)
    #doy = self._day_of_year().reshape(1, -1)
    moy = self._month_of_year().reshape(1, -1)
    woy = self._week_of_year().reshape(1, -1)

    all_covs = [
        moh,
        hod,
        dom,
        dow,
        #doy,
        moy,
        woy,
    ]
    columns = ["moh", "hod", "dom", "dow", "moy", "woy"]
    if self.use_holiday_distance:
      hol_covs = self._get_distance_holidays()
      all_covs.append(hol_covs)
      columns += [f"hol_{i+1}" for i in range(len(HOLIDAYS))]

    if self.use_which_holiday:
      hol_covs = self._get_which_holiday()
      all_covs.append(hol_covs)
      columns += ["hol"]
    
    print(hol_covs.shape, woy.shape, moh.shape)
    return pd.DataFrame(
        data=np.vstack(all_covs).transpose(),
        columns=columns,
        index=self.dti,
    )

if __name__=="__main__":
  
  date = pd.date_range(start='2023-01-01 00:00', end='2023-1-31 23:59', freq='H')
  
  time = TimeCovariates(date,
                        use_holiday_distance=False, 
                        use_which_holiday=True)
  time_features = time.get_covariates()
  print(time_features)