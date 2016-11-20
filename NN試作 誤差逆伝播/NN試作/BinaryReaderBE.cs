using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing.Imaging;
using System.Threading.Tasks;
using System.IO;

namespace NN試作
{
    class BinaryReaderBE : BinaryReader
    {
        public BinaryReaderBE(Stream input)
            : base(input)
        {
        }
        public BinaryReaderBE(Stream input, Encoding encoding)
            : base(input, encoding)
        {
        }

        public override short ReadInt16()
        {
            return _ToBigEndian(base.ReadInt16());
        }
        public override int ReadInt32()
        {
            return _ToBigEndian(base.ReadInt32());
        }
        public override long ReadInt64()
        {
            return _ToBigEndian(base.ReadInt64());
        }
        public override ushort ReadUInt16()
        {
            return _ToBigEndian(base.ReadUInt16());
        }
        public override uint ReadUInt32()
        {
            return _ToBigEndian(base.ReadUInt32());
        }
        public override ulong ReadUInt64()
        {
            return _ToBigEndian(base.ReadUInt64());
        }
        public override float ReadSingle()
        {
            return _ToBigEndian(base.ReadSingle());
        }
        public override double ReadDouble()
        {
            return _ToBigEndian(base.ReadDouble());
        }
        public override decimal ReadDecimal()
        {
            throw new NotImplementedException();
        }

        private short _ToBigEndian(short value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToInt16(bytes, 0);
        }

        private ushort _ToBigEndian(ushort value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToUInt16(bytes, 0);
        }

        private int _ToBigEndian(int value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        private uint _ToBigEndian(uint value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToUInt32(bytes, 0);
        }

        private long _ToBigEndian(long value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToInt64(bytes, 0);
        }

        private ulong _ToBigEndian(ulong value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToUInt64(bytes, 0);
        }

        private float _ToBigEndian(float value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToSingle(bytes, 0);
        }

        private double _ToBigEndian(double value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            bytes = _ReverseBytes(bytes);
            return BitConverter.ToDouble(bytes, 0);
        }

        private byte[] _ReverseBytes(byte[] bytes)
        {
            if (bytes == null)
            {
                return null;
            }
            return bytes.Reverse().ToArray();
        }
    }
}
