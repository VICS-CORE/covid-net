from src import constants
from src import data
from src import utils
from src import predictions
import datetime as dt
import argparse

parser = argparse.ArgumentParser(prog='gen_statewise_predictions', description="Generate statewise predictions")
parser.add_argument('experimentid', help='experiment id e.g. 0011')
parser.add_argument('checkpoint', help='checkpoint filename e.g. latest-e1000.pt')
parser.add_argument('-d', '--days', type=int, default=120, help='number of days to predict. default 120')
parser.add_argument('-o', '--offset', type=int, default=1, help='number of days of input data to skip. default 1')
parser.add_argument('-t', '--taskidx', type=int, choices=[0, 1], default=0, help='task index to predict e.g. 0 for confirmed, 1 for deaths. default 0')
parser.add_argument('-vpf', '--vpfile', default='', help='name of videoplayer predictions file.')
parser.add_argument('-trf', '--trfile', default='', help='name of tracker predictions file.')
args = parser.parse_args()

assert(args.vpfile or args.trfile)
assert(args.offset>=1)

states_df = data.get_statewise_data()
model, cp = utils.load_model(args.experimentid, args.checkpoint, v=False)
prediction_date = (states_df.date.max().to_pydatetime() - dt.timedelta(days=args.offset)).strftime("%Y-%m-%d")
print("Predicting for:", prediction_date)

api = predictions.generate(
    states_df,
    constants.STT_INFO,
    model,
    cp,
    args.taskidx,
    args.days,
    args.offset,
    plot=False
)

if args.vpfile:
    predictions.export_videoplayer(api, prediction_date, args.vpfile)
if args.trfile:
    predictions.export_tracker(api, args.trfile)

print("Done")
