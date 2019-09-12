import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import java.io._
import scala.io.Source
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ListBuffer

val datafile = "human_stitch5.csv"
val file = sc.textFile(datafile)
val data = file.map(line => {val vals = line.split(","); (vals(0).toInt,vals(1).toInt,vals(2).toDouble)}).persist()
val sparseMatrix = new CoordinateMatrix(data.map { case (row, col, data) => MatrixEntry(row, col, data) })
val A = sparseMatrix.toIndexedRowMatrix()
val svd = A.computeSVD(200,computeU = true)
val a = svd.s.toArray
a.foreach(x=>println(x))
