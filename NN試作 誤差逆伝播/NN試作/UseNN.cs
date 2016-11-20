using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
namespace NN試作
{
    public class UseNN
    {
        public List<int> UnitCounts = new List<int>();//ユニット数
        public List<int> LowTypes = new List<int>();//層の種類
        public List<int> FilterCounts = new List<int>();//フィルタ数
        public int DefaultChannels;//画像のチャネル数
        public List<DenseVector[]> Trains = new List<DenseVector[]>();//訓練データ
        public List<DenseVector[]> Tests = new List<DenseVector[]>();//テストデータ
        public int MiniBatchSampleCount = 1;//ミニバッチに使うサンプルの数
        public int MiniBatchCount = 10000;//ミニバッチ自体の数
        public List<DenseVector[][]> MiniBatchs = new List<DenseVector[][]>();


        public UseNN() { }
        public UseNN(IEnumerable<int> unitcounts, IEnumerable<int> lowtypes)
        {
            int ucount = unitcounts.Count(), lcount = lowtypes.Count();
            if (lcount > ucount) { throw new NullReferenceException("ユニットのタイプに対してユニットの数のほうが大きいとは何事か"); }
            if (ucount > lcount) { lowtypes.Concat(new int[ucount - lcount]); }
            UnitCounts.AddRange(unitcounts);
            LowTypes.AddRange(lowtypes);
            FilterCounts.AddRange(new List<int>(ucount));//とりあえず0埋め
        }
        public UseNN(int unit, int lowtype) { UnitCounts.Add(unit); LowTypes.Add(lowtype); FilterCounts.Add(0); }

        public void Add(IEnumerable<int> unitcounts, IEnumerable<int> lowtypes,IEnumerable<int>filters=null)
        {
            int ucount = unitcounts.Count(), lcount = lowtypes.Count();
            if (lcount > ucount) { throw new NullReferenceException("ユニットのタイプに対してユニットの数のほうが大きいとは何事か"); }
            if (ucount > lcount) { lowtypes.Concat(new int[ucount - lcount]); }
            UnitCounts.AddRange(unitcounts);
            LowTypes.AddRange(lowtypes);
            if (filters != null) { FilterCounts.AddRange(filters); }
        }
        public void Add(int unit, int lowtype, int filter = 0) { UnitCounts.Add(unit); LowTypes.Add(lowtype); FilterCounts.Add(filter); }

        public void AddByFilter(int channel, int filter, int filtersize, int stride, int pad)//畳み込み層を追加
        {
            
        }









        public NN NetWork;//ネットワーク
        public void BatchInit(int minibatchcount,int minibatchsamplecount)
        {
            MiniBatchCount = minibatchcount;MiniBatchSampleCount = minibatchsamplecount;
        }
        public void DataInit(List<DenseVector[]> trains, List<DenseVector[]> tests)
        {
            Trains = trains.ToList();Tests = tests.ToList();
        }
        public void NNinit()
        {
            NetWork = new NN(UnitCounts.Count);
            NetWork.Initialize(UnitCounts.ToArray(),LowTypes.ToArray());
            if (MiniBatchCount != -1)
            {
                for (int i = 0; i < MiniBatchCount; i++)
                {
                    MiniBatchs.Add(NetWork.RandomSelect(Trains, MiniBatchSampleCount).ToArray());
                }
            }
        }
        public void Train()
        {
            if (MiniBatchCount == -1)
            {
                foreach (var i in Trains)
                {
                    NetWork.NormaTrainl(i);
                }
            }
            else
            {
                foreach (var i in MiniBatchs)
                {
                    NetWork.MiniBatch(i);
                }
            }
        }
        public void Test()
        {
            for (int i = 0; i < Tests.Count; i++)
            {
                var output = NetWork.ForwardPropagation(Tests[i]);
                Console.WriteLine(Tests[i][0][0] + ":" + Tests[i][1][0] + "=" + output[0]);
            }
        }
        
    }
}
