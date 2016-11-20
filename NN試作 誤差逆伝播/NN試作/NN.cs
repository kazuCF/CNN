using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
namespace NN試作
{
    public class NN
    {
        enum Low
        {
            ketugou, conv, pooling, normalize
        }
        public int LowCount;
        public DenseMatrix[] Weight;
        public DenseVector[] Bias;
        public DenseVector[] U;
        public DenseVector[] Z;
        public List<DenseVector> Filters;

        public List<DenseVector[]> Tr;
        public int[] UnitCounts;
        public double Eps = 0.2;
        public DenseVector D;
        //ここからは後から
        public DenseVector[] Delta;
        //層の種類を決定...畳みこみとか
        public int[] LowTypes;
        public int[] FilterCount;//フィルタ数(所謂m)一つ前の層のmが、その層にとってのkとなる。
        public int DefaultChannel;//最初のチャネル数。モノクロなら1、カラーなら3
        //ミニバッチ用の変数

        public DenseVector[] Ds;
        public DenseVector[] Ys;
        public DenseVector[][] Zs;
        public DenseVector[][] Deltas;

        public NN(int lowcount)
        {
            LowCount = lowcount;
            Weight = new DenseMatrix[LowCount];
            Tr = new List<DenseVector[]>();
            Bias = new DenseVector[LowCount];
            U = new DenseVector[LowCount];
            Z = new DenseVector[LowCount];
            Delta = new DenseVector[LowCount];
            UnitCounts = new int[LowCount];
        }
        public void Initialize(int[] unitcounts,int[] lowtypes)
        {
            int filterheight = 5;
            Random Rnd = new Random();
            unitcounts.CopyTo(UnitCounts, 0);
            lowtypes.CopyTo(LowTypes, 0);
            for (int i = 0; i < LowCount; i++)
            {
                if (LowTypes[i] == 1)//畳みこみの場合
                {
                    Tr.Add(new DenseVector[784]);
                    DenseVector[] filter=new DenseVector[20];//フィルター
                    for (int count = 0; count < 20; count++)
                    {
                        for (int j = 0; j < filterheight; j++)
                        {
                            for (int k = 0; k < filterheight; k++)
                            {
                                filter[count][j * filterheight + k] = Rnd.Next(255);
                            }
                        }
                    }

                        for (int k = 0; k < unitcounts[i]; k++)
                        {
                            for (int j = 0; j < unitcounts[i - 1]; j++)
                            {
                                int jN = j / 784;
                                int jx = (j % 784) % 28;
                                int jy = (j % 784) / 28;
                                int kN = k / 784;
                                int kx = (k % 784) % 28;
                                int ky = (k % 784) / 28;
                                if (kx - jx >= 0 && kx - jx <= 4 && ky - jy >= 0 && ky - jy <= 4)
                                {
                                    Weight[i][k, j] = filter[kN][(ky-jy+2)*filterheight+(kx-jx+2)];
                                    Tr[i][kN * 784 + (ky - jy + 2) * filterheight + (kx - jx + 2)][j] = 1;
                                }
                                if (jy - (ky - jy + 2) < 0 || jy - (ky - jy + 2) >= 28 || jx - (kx - jx + 2) < 0 || jx - (kx - jx + 2) >= 28)
                                {
                                    Weight[i][k, j] = 0;
                                    Tr[i][kN * 784 + (ky - jy + 2) * filterheight + (kx - jx + 2)][j] = 0;
                                }
                            }
                        }
                    //Filters.Add(filter);
                }
                if (i > 0)
                {
                    Weight[i] = new DenseMatrix(unitcounts[i], unitcounts[i - 1]);
                    for (int k = 0; k < unitcounts[i]; k++)
                    {
                        for (int j = 0; j < unitcounts[i - 1]; j++)
                        {
                            if (LowTypes[i] != 1) Weight[i][k, j] = Math.Sign(Rnd.NextDouble() - 0.5) * Rnd.NextDouble();
                        }
                    }

                }
                Bias[i] = new DenseVector(unitcounts[i]);
                for (int k = 0; k < unitcounts[i]; k++)
                {
                    //
                    Bias[i][k] = Math.Sign(Rnd.NextDouble() - 0.5) * Rnd.NextDouble();
                    // Bias[i][k] = 1;
                }
                U[i] = new DenseVector(unitcounts[i]);
                Z[i] = new DenseVector(unitcounts[i]);
                Delta[i] = new DenseVector(unitcounts[i]);
            }
        }
        public void Initialize(DenseMatrix[] weight)
        {
            Weight = Copy(weight);

        }
        public void NormaTrainl(DenseVector[]xd)
        {
            MiniBatch(new DenseVector[][] { xd });
        }
        public void MiniBatch(DenseVector[][] xds)
        {
            DenseMatrix[] dw = new DenseMatrix[Weight.Count()];
            DenseVector[] db = new DenseVector[LowCount];
            dw = Copy(Weight);
            db = Copy(Bias);
            zerofill(ref dw);
            zerofill(ref db);
            int count = xds.Count();
            for (int i = 0; i < count; i++)
            {
                ForwardPropagation(xds[i]);
                WeBias we = BackPropagation();
                dw=Add(dw, we.W);
                db=Add(db, we.B);
            }
            dw = Multi(dw, 1.0 / count);
            db = Multi(db, 1.0 / count);
            DenseVector[] epb = Multi(db, Eps);
            Bias = Minus(Bias, epb);
            DenseMatrix[] epw = Multi(dw, Eps);
            Weight = Minus(Weight, epw);
        }
     /*   public void ForwardPropagation(DenseVector[][] xds)
        {
            for (int i = 0; i < xds.Count(); i++)
            {
                Ds[i] = xds[i][1];
                Ys[i] = ForwardPropagation(xds[i]);

            }
        }*/


        public DenseVector ForwardPropagation(DenseVector[] xd)
        {
            Z[0] = Copy(xd[0]); D = Copy(xd[1]);
            for (int i = 1; i < LowCount; i++)
            {
                U[i] = Weight[i] * Z[i - 1] + Bias[i];
                // Z[i] = Logisitics(U[i]);
                Z[i] = Tanh(U[i]);//
                // Z[i] = Copy(U[i]);
            }
            Z[LowCount - 1] = Copy(U[LowCount - 1]);//出力層に対してはTanhを用いない
            return Z[LowCount - 1];
        }

        public WeBias BackPropagation()//確率的降下法
        {
            Delta[LowCount - 1] = Z[LowCount - 1] - D;
            DenseMatrix[] dw = new DenseMatrix[Weight.Count()];
            dw = Copy(Weight);
            DenseVector[] db = new DenseVector[LowCount];
            db = Copy(Bias);
            List<DenseVector> dh = new List<DenseVector>(); //畳みこみのフィルタ用
            dh = Copy(Filters);
            //   Bias.CopyTo(db, 0);
            for (int i = LowCount - 2; i >= 1; i--)
            {
                //  Delta[i] = Multi(DLogisitics(U[i]), (Weight[i + 1].Transpose() * Delta[i + 1]));
                // Delta[i] = ToDense(Weight[i + 1].Transpose() * Delta[i + 1]);
                switch (LowTypes[i])
                {
                    case 0://全結合層
                        Delta[i] = Multi(DTanh(U[i]), (Weight[i + 1].Transpose() * Delta[i + 1]));
                        break;
                    case 1://畳み込み層
                        Delta[i] = Multi(DTanh(U[i]), (Weight[i + 1].Transpose() * Delta[i + 1]));
                        
                            //Dot(dw[i], Tr[i][0]);
                        DenseVector dwvec=MatToVec(dw[i]);
                       
                        for (int r = 0; r < dh[i].Count; r++)
                        {
                            dh[i][r] = Dot(dwvec, Tr[i][r]);
                            //DenseVector PartTr = PartialVector(Tr[i][r], r * 784, (r + 1) * 784 - 1);
                        }
                        break;
                    case 2://プーリング層
                        break;
                    case 3://正規化層

                        break;
                    default:
                        break;
                }


            }

            for (int l = 1; l < LowCount; l++)
            {
                //  Delta[l].CopyTo(db[l]);
                db[l] = Copy(Delta[l]);
                for (int j = 0; j < dw[l].RowCount; j++)
                {
                    for (int i = 0; i < dw[l].ColumnCount; i++)
                    {
                        dw[l][j, i] = Delta[l][j] * Z[l - 1][i];
                    }
                }
                // Bias[l] = Bias[l] - Eps * db[l];
                //Weight[l] = Weight[l] - Eps * dw[l];
            }
            return new WeBias(dw, db);
        }
        public DenseVector PartialVector(DenseVector a, int n1, int n2)
        {
            DenseVector b = new DenseVector(n2 - n1 + 1);
            for (int i = 0; i < b.Count; i++)
            {
                b[i] = a[i + n1];
            }
            return b;
        }
      /*  public double Dot(DenseMatrix Mat, DenseVector VecA)
        {
            double x=0;
            for (int i = 0; i < VecA.Length; i++)
            {
                for (int k = 0; k < VecA[i].Count; k++)
                {
                    x += Mat[i, k] * VecA[i][k];
                }
            }
            return x;
        }*/
        public double Tanh(double x)
        {
            return Math.Tanh(x);
        }
        public DenseVector Tanh(DenseVector vec)
        {
            DenseVector vec2 = new DenseVector(vec.Count);
            vec2 = Copy(vec);
            for (int i = 0; i < vec2.Count; i++)
            {
                vec2[i] = Tanh(vec[i]);
            }
            return vec2;
        }
        public double DTanh(double x)
        {
            return 1 - Math.Tanh(x) * Math.Tanh(x);
        }
        public DenseVector DTanh(DenseVector vec)
        {
            DenseVector vec2 = new DenseVector(vec.Count);
            vec2 = Copy(vec);
            for (int i = 0; i < vec2.Count; i++)
            {
                vec2[i] = DTanh(vec[i]);
            }
            return vec2;
        }
        /*       public DenseVector Logisitics(DenseVector vec)
               {
                   DenseVector vec2=new DenseVector(vec.Count);
                   vec2 = Copy(vec);
                   for (int i = 0; i < vec2.Count; i++)
                   {
                       vec2[i] = Logisitics(vec[i]);
                   }
                   return vec2;
               }
               public double Logisitics(double x)backp
               {
                   double a= 1 / (1 + Math.Exp(-x));
                   return a;
               }
               public DenseVector DLogisitics(DenseVector vec)
               {
                   DenseVector vec2 = new DenseVector(vec.Count);
                   vec2 = Copy(vec);
                   for (int i = 0; i < vec2.Count; i++)
                   {
                       vec2[i] = DLogisitics(vec[i]);
                   }
                   return vec2;
               }
               public double DLogisitics(double x)//第一導関数
               {
                   return Logisitics(x) * (1 - Logisitics(x));
               }*/
        public DenseMatrix Multi(DenseMatrix a, DenseMatrix b)
        {
            DenseMatrix c = new DenseMatrix(a.RowCount, a.ColumnCount);
            for (int i = 0; i < a.RowCount; i++)
            {
                for (int k = 0; k < a.ColumnCount; k++)
                {
                    c[i, k] = a[i, k] * b[i, k];
                }
            }
            return c;
        }

        public DenseVector Multi(MathNet.Numerics.LinearAlgebra.Vector<double> a, MathNet.Numerics.LinearAlgebra.Vector<double> b)
        {
            DenseVector c = new DenseVector(a.Count);
            for (int i = 0; i < a.Count; i++)
            {
                c[i] = a[i] * b[i];
            }
            return c;
        }
        public DenseVector ToDense(MathNet.Numerics.LinearAlgebra.Vector<double> a)
        {
            DenseVector c = new DenseVector(a.Count);
            for (int i = 0; i < a.Count; i++)
            {
                c[i] = a[i];
            }
            return c;
        }
        public DenseVector Copy(DenseVector vec)
        {
            DenseVector c = new DenseVector(vec.Count);
            for (int i = 0; i < vec.Count; i++)
            {
                c[i] = vec[i];
            }
            return c;
        }
        public DenseVector[] Copy(DenseVector[] mat)
        {
            DenseVector[] c = new DenseVector[mat.Count()];
            for (int i = 0; i < mat.Count(); i++)
            {
                c[i] = Copy(mat[i]);
            }
            return c;
        }
        public List<DenseVector> Copy(List<DenseVector> mat)
        {
            List<DenseVector> c = new List<DenseVector>();
            for (int i = 0; i < mat.Count(); i++)
            {
                c[i].Add(Copy(mat[i]));
            }
            return c;
        }
        public DenseMatrix Copy(DenseMatrix mat)
        {
            if (mat == null) return null;
            DenseMatrix c = new DenseMatrix(mat.RowCount, mat.ColumnCount);
            for (int i = 0; i < c.RowCount; i++)
            {
                for (int j = 0; j < c.ColumnCount; j++)
                {
                    c[i, j] = mat[i, j];
                }
            }
            return c;
        }
        public DenseMatrix[] Copy(DenseMatrix[] mat)
        {
            DenseMatrix[] c = new DenseMatrix[mat.Count()];
            for (int i = 0; i < mat.Count(); i++)
            {
                c[i] = Copy(mat[i]);
            }
            return c;
        }
        public List<DenseVector[]> Copy(List<DenseVector[]> vecl)
        {
            List<DenseVector[]> c = new List<DenseVector[]>();
            for (int i = 0; i < vecl.Count; i++)
            {
                c[i] = Copy(vecl[i]);
            }
            return c;
        }

        public DenseVector[] Add(DenseVector[] x, DenseVector[] y)//要素同士の
        {
            int count = x.Count();
            DenseVector[] z = new DenseVector[count];
            for (int i = 0; i < count; i++)
            {
                z[i] = x[i] + y[i];
            }
            return z;
        }
        public DenseVector[] Minus(DenseVector[] x, DenseVector[] y)//要素同士の
        {
            int count = x.Count();
            DenseVector[] z = new DenseVector[count];
            for (int i = 0; i < count; i++)
            {
                z[i] = x[i] - y[i];
            }
            return z;
        }
        public DenseVector[] Multi(DenseVector[] x, double y)
        {
            int count = x.Count();
            DenseVector[] z = new DenseVector[count];
            for (int i = 0; i < count; i++)
            {
                z[i] = x[i] * y;
            }
            return z;
        }
        public DenseMatrix[] Add(DenseMatrix[] x, DenseMatrix[] y)
        {
            int count = x.Count();
            DenseMatrix[] z = new DenseMatrix[count];
            for (int i = 0; i < count; i++)
            {
                if (x[i] == null || y[i] == null) { continue; }
                z[i] = x[i] + y[i];
            }
            return z;
        }
        public DenseMatrix[] Minus(DenseMatrix[] x, DenseMatrix[] y)
        {
            int count = x.Count();
            DenseMatrix[] z = new DenseMatrix[count];
            for (int i = 0; i < count; i++)
            {
                if (x[i] == null || y[i] == null) { continue; }
                z[i] = x[i] - y[i];
            }
            return z;
        }
        public DenseMatrix[] Multi(DenseMatrix[] x,double y)
        {
            int count = x.Count();
            DenseMatrix[] z = new DenseMatrix[count];
            for (int i = 0; i < count; i++)
            {
                if (x[i] == null) { continue; }
                z[i] = x[i] * y;
            }
            return z;
        } 
        public void zerofill(ref DenseMatrix[]x)
        {
            int count = x.Count();
            for (int i = 0; i < count; i++)
            {
                if (x[i] == null) continue;
                x[i].Clear();
            }
        }
        public void zerofill(ref DenseVector[]x)
        {
            int count = x.Count();
            for (int i = 0; i < count; i++)
            {
                x[i].Clear();
            }
        }
        public IEnumerable<T> RandomSelect<T>(IEnumerable<T>a,int count)
        {
            var alist = a.ToList();
            var result = new List<T>();
            Random rnd = new Random();
            for (int i = 0; i < count; i++)
            {
                int index = rnd.Next(alist.Count());
                result.Add(alist[index]);
                alist.RemoveAt(index);
            }
            return result;
        }
        public DenseVector MatToVec(DenseMatrix Mat)
        {
            DenseVector vec = new DenseVector(Mat.RowCount * Mat.ColumnCount);
            for (int i = 0; i < Mat.RowCount; i++)
            {
                for (int k = 0; k < Mat.ColumnCount; k++)
                {
                    vec[i * Mat.RowCount + k] = Mat[i, k];   
                }
            }
            return vec;
        }
        public double Dot(DenseVector a, DenseVector b)
        {
            double x = 0;
            for (int i = 0; i < a.Count; i++)
            {
                x += a[i] * b[i];
            }
            return x;
        }
    }
    public class WeBias
    {
       public DenseMatrix[] W;
       public DenseVector[] B;
        public WeBias(DenseMatrix[] w,DenseVector[] b)
        {
            W = w;B = b;
        }
    }
}
