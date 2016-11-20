using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;


namespace NN試作
{
    class MNistLabelLoader
    {
        public int magicNumber;
        public int numberOfItems;
        public byte[] labelList;
        public MNistLabelLoader()
        {
        }

        public static MNistLabelLoader Load(string path)
        {
            if (File.Exists(path) == false)
            {
                return null;
            }
            FileStream stream = new FileStream(path, FileMode.Open);
            BinaryReaderBE reader = new BinaryReaderBE(stream);
            MNistLabelLoader loader = new MNistLabelLoader();
            loader.magicNumber = reader.ReadInt32();
            loader.numberOfItems = reader.ReadInt32();
            loader.labelList = reader.ReadBytes(loader.numberOfItems);
            reader.Close();
            return loader;
        }
    }
}
