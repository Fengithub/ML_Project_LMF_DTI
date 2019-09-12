import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.util.AccumulatorV2
import collection.JavaConverters._
import scala.io.Source
import org.apache.spark.rdd._
import scala.util.Random
import java.util.Calendar
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import java.io._
import scala.math._

 
   

    def addToArr(v:  Array[Double], d: (Int, Double)):  Array[Double] = {
        v(d._1) = d._2
        v
    }

    def combineArr  (s: Array[Double], v: Array[Double] ): Array[Double]  = {
        for(i <- 0 until s.length){
            s(i) += v(i)
        }
        return s
    }
    
        //compute and return s += v
    def add(s: Array[Array[Double]], v: Array[Array[Double]] ): Array[Array[Double]]  = {
        for(i <- 0 until s.length){
            for(j <- 0 until s(i).length) {
                s(i)(j) += v(i)(j)
            }
        }
        return s
    }
    
    def div(s: Array[Array[Double]], d: Double ): Array[Array[Double]]  = {
        for(i <- 0 until s.length){
            for(j <- 0 until s(i).length) {
                s(i)(j) /= d
            }
        }
        return s
    }

    def sumarr(s: Array[Array[Double]] ): Double = {
        var total = 0.0
        for(i <- 0 until s.length){
            for(j <- 0 until s(i).length) {
                total += s(i)(j)
            }
        }
        return total
    }
    def randarr(K: Int) : Array[Double] = {
        var ret = Array.fill(K)(0.0)
        val r = scala.util.Random
        for(i <- 0 until K) {
            ret(i) = r.nextFloat
        }
        ret
    }


//---------------------------- k-fold cross validation --------------------------


def tenFoldCrossValidation (data: org.apache.spark.rdd.RDD[(Int, Int, Double)]) = 
{
	for (i <- 0 until 10) yield
	{
		var cnt = 0
		val test = data.filter {j => cnt+=1; cnt % 10 == i}
                cnt = 0
		val train = data.filter {j => cnt+=1; cnt % 10 != i}
		(test, train)
	}
	
}



//--------------------------------------------------------------------------------

    

        val datafile = "human_stitch4.csv"
	val outfile = "human.out"

        
        val ofile = new File(outfile)
        val output = new BufferedWriter(new FileWriter(ofile))
        
        var conf = new SparkConf().setAppName("Assign2")
        var sc = new SparkContext(conf)
        
        val file = sc.textFile(datafile)

        val data = file.map(line => {val vals = line.split(","); (vals(0).toInt,vals(1).toInt,vals(2).toDouble)}).persist()
        val ncols = data.reduce( (a,b) => if (a._2 > b._2) a else b )._2+1        
        val nrows = data.reduce( (a,b) => if (a._1 > b._1) a else b )._1+1 
	//val ncols = 9931 
	//val nrows = 558947
        // TO DO: perform 10 fold cross validation, split the data into 10 folds, each time set 9 as training data, 1 as validation data  
	//data.repartition(26)
	val crossVal = tenFoldCrossValidation (data)
	val allRes = scala.collection.mutable.ListBuffer.empty[(Double, Double, Double,Double,Double)]

        val MAX = 24
	//val MAX = 2
        val alpha = 0.01        
        val beta = 0.0005                
        val gamma = 0.1                        
        val K = 50
        val start = System.nanoTime()
	//val validation_data = crossVal(0)._1
	//val train_data = crossVal(0)._2

	for ( (validation_data,train_data) <- crossVal)
	{//split by rows
        	var rowdata = train_data.map(x => (x._1,(x._2,x._3))).groupByKey().map(x => (x._1,(x._2, randarr(K)))).persist()
        	var H = Array.fill(ncols)(randarr(K))
                
        	for (i <- 0 until MAX) 
		{
            		val a = alpha/(i*gamma+1) //endless parameter tweaking..
            		val Hb = sc.broadcast(H)
            		val stepres = rowdata.mapPartitions( iter => 
			{
                		//gradient update
                		val H = Hb.value
                		var loss = 0.0
                		var res = collection.mutable.ArrayBuffer.fill(1)((0.0, 1, H))
                		var myiter = iter
                		for ( b <- 0 until 40 ) {
                    			loss = 0.0
                    			val (it, it2) = myiter.duplicate
                    			myiter = it2
                    			it.foreach(rec => 
					{ //for each row
                       				val r = rec._1
                       		 		var W = rec._2._2
                        			rec._2._1.foreach( x => 
							{ //for each column
                            					val c = x._1
                            					val v = x._2
                            					var Hc = H(c)
                            					//compute dot product of row of W and column of H
                            					var dot = 0.0
                            					for (k <- 0 until K) 
								{
                               		 				dot += W(k)*Hc(k)
                            					}
                                                val p = exp(dot)/(1+exp(dot))
                            
                            				//error
                            					val eij = v - p
                            					for (k <- 0 until K) {
                                					val w = W(k)
                                					val h = Hc(k)
                                					W(k) += a * (eij * h - beta*w)
                                					Hc(k) += a * (eij * w - beta*h)
                            					}
                            
                            					loss += eij*eij
                            
                        				})
                    			})
                		}
                		res(0) = res(0).copy(_1 = loss)
                		res.iterator
            		}).persist()
            
            		val res = stepres.reduce( (a,b) => (a._1+b._1, a._2+b._2, add(a._3,b._3)) )  //problems
            		H = div(res._3,res._2)
            		//println(res._1)
        	}
        
        	val Wrows = new IndexedRowMatrix(rowdata.map(r => new IndexedRow(r._1,new DenseVector(r._2._2))).persist())
        
	// TO DO: Modify this part, calculate the scores of validation data set, output:
	// RMSD
	//set cutoff value = 0.7, calculate precision, recall enhancement

        	val missing = validation_data.map(r => (r._1,(r._2, r._3 ) ) ).groupByKey.collectAsMap
        
        	val Hb = sc.broadcast(H)
        	val missingb = sc.broadcast(missing)
        	val outvals = rowdata.mapPartitions(iter => 
		{
                	//gradient update
                	val H = Hb.value
                	val missing = missingb.value
                	var res = collection.mutable.ArrayBuffer.empty[(Int,Int,Double, Double, Double)]
                	iter.foreach(rec => 
			{ //for each row
                        	val r = rec._1
                        	var W = rec._2._2
                        	if( missing.contains(r) ) 
				{
                            		missing(r).foreach
					{case (col,v) => 
					{
                                		var dot = 0.0
                                		var Hc = H(col)

                                		for (k <- 0 until K) 
						{
                                    			dot += W(k)*Hc(k)
                                		}
                        val p = exp(dot)/(1+exp(dot))
						val rdm = scala.util.Random
                               	 		res.append( (r,col, v, p, rdm.nextDouble) )
                            		}}
                        	}
                	})
                	res.iterator
            	})

		
		var diff = 0.0	
		val ot = outvals.collect
		
		val ct = ot.size
		ot.foreach {case(r,c,v,e,rdm) => diff += (v-e)*(v-e)}
		val sqdf = diff / ct
		
		val vnum = ot.filter {case (r,c,v,e,rdm) => v > 0.7}.size
		val venum = ot.filter {case (r,c,v,e,rdm) => v > 0.7 && e > 0.7}.size
		val vrnum = ot.filter {case (r,c,v,e,rdm) => v > 0.7 && rdm > 0.7}.size
		val enum = ot.filter {case (r,c,v,e,rdm) => e > 0.7}.size
		val rnum = ot.filter {case (r,c,v,e,rdm) => rdm > 0.7}.size

                val pres = (venum*1.0) / enum 
		val pres_enr = ((venum*1.0) / enum ) / ((vrnum*1.0)/rnum)
                val recall = (venum*1.0) / vnum 
		val rec_enr = ((venum*1.0) / vnum ) /  ((vrnum*1.0)/vnum)

		allRes += ((sqdf, pres, pres_enr, recall,rec_enr))
		
        

		
	}
	
	for (x <- allRes){
        output.write(x._1+","+x._2+","+x._3+","+x._4+","+x._5+"\n")
        }
        output.close()
	    
        

        
        //println("#Elapsed time: " + (System.nanoTime()-start)/1000000000.0)



