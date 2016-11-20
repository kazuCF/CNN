using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
namespace NN試作
{
    class Program
    {
        static void Use()
        {

            //////////データ用意
            Random rnd = new Random();
            List<DenseVector[]> xds = new List<DenseVector[]>();
            for (int i = 0; i < 1000000; i++)
            {
                xds.Add(new DenseVector[2] { new DenseVector(1), new DenseVector(1) });
                double a = rnd.NextDouble();
                xds[i][0][0] = a; xds[i][1][0] = a * a;
            }

            List<DenseVector[]> Tests = new List<DenseVector[]>();
            for (int i = 0; i < 100; i++)
            {
                Tests.Add(new DenseVector[2] { new DenseVector(1), new DenseVector(1) });
                double a = rnd.NextDouble();
                Tests[i][0][0] = a; Tests[i][1][0] = a * a;
            }
            //////////データ用意終


            UseNN test = new UseNN();
            test.Add(new int[] { 1, 3, 1 },new int[] { 0, 0, 0 });
            test.BatchInit(10000, 10);
            test.DataInit(xds, Tests);
            test.NNinit();
            test.Train();
            test.Test();
            Console.ReadLine();
        }
        static void ZikoHugouka(List<DenseVector>Me)
        {
            UseNN Ziko = new UseNN();
            Ziko.Add(new int[] { Me[0].Count, Me[0].Count / 2, Me[0].Count}, new int[] { 0, 0, 0 });
            List<DenseVector[]> xds = new List<DenseVector[]>();
            for (int i = 0; i < Me.Count; i++)
            {
                xds.Add(new DenseVector[] { Me[i], Me[i] });//inputとoutputは同じが理想
            }
            Ziko.BatchInit(-1, 1);//バッチ処理は利用しない
            Ziko.DataInit(xds, xds);//テスト集合と訓練集合は同一
            Ziko.NNinit();
            Ziko.Train();      
        }
        static void CNN(List<DenseVector[]> Data)//畳み込みニューラルネットワーク
        {
             //MNISTの読み込み
            MNistImageLoader imgLoader = MNistImageLoader.Load("train-images.idx3-ubyte");
            MNistLabelLoader lLoader = MNistLabelLoader.Load("train-labels.idx1-ubyte");
            //////////データ用意
            Random rnd = new Random();
            List<DenseVector[]> xds = new List<DenseVector[]>();
            for (int i = 0; i < 10000000 ; i++)
            {
                xds.Add(new DenseVector[2] { new DenseVector(1), new DenseVector(1) });
                double a = rnd.NextDouble();
                xds[i][0][0] = a; xds[i][1][0] = 5 * a * a;
            }

            List<DenseVector[]> Tests = new List<DenseVector[]>();
            for (int i = 0; i < 100; i++)
            {
                Tests.Add(new DenseVector[2] { new DenseVector(1), new DenseVector(1) });
                double a = rnd.NextDouble();
                Tests[i][0][0] = a; Tests[i][1][0] = a * a;
            }
            //////////データ用意終

            UseNN UseCNN = new UseNN();
            UseCNN.Add(new int[] { imgLoader.numberOfRows * imgLoader.numberOfColumns, 100, 100, 100, 100, 100, 100, 20, 20, 9 }
                , new int[] { 0, 1, 2, 1, 2, 1, 2, 0, 0, 0 });

        }
        static void Main(string[] args)
        {
            //MNISTの読み込み
            MNistImageLoader imgLoader = MNistImageLoader.Load("train-images.idx3-ubyte");
            MNistLabelLoader lLoader = MNistLabelLoader.Load("train-labels.idx1-ubyte");
            Console.WriteLine(imgLoader.numberOfImages);
            Console.WriteLine(lLoader.numberOfItems);
            int n = 3375;
            byte[] image = imgLoader.bitmapList[n];
            byte label = lLoader.labelList[n];
            for (int i = 0; i <imgLoader.numberOfRows; i++)
            {
                for (int j = 0; j < imgLoader.numberOfColumns; j++)
                {
                    if (image[i * imgLoader.numberOfRows + j] >= 255)
                    {
                        Console.Write("#");
                    }
                    else Console.Write(" ");
                }
                Console.WriteLine();
            }
            Console.WriteLine(label);
            Console.ReadLine();

        }


    }
}
