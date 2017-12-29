package org.yli.learn;

import com.google.common.base.Stopwatch;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.concurrent.TimeUnit;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

/**
 * Hello world!
 *
 */
public class App {
  public static void main( String[] args ) {
    INDArray nd = Nd4j.create(new float[] {1, 2, 3, 4}, new int[] {2, 2});
    System.out.println(nd);

    nd = nd.transpose();
    System.out.println(nd);

//    DoubleArray nd2 =
//        Nd4j.create(
//            new double[] {1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0},

    System.out.println(Nd4j.zeros(10));

    System.out.println(Nd4j.zeros(3, 5).addi(10));

    System.out.println(Nd4j.ones(4, 4));

    INDArray nd2 = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[]{2, 6});
    System.out.println(nd2);

    System.out.println(nd2.reshape(3, 4));
    System.out.println(nd2);

    nd2 = nd2.reshape(3, 4);
    System.out.println(nd2);

    System.out.println(nd2.linearView());

    INDArray ndv = sigmoid(nd2);
    System.out.println(ndv);

    System.out.println(Transforms.tanh(Nd4j.rand(3, 4)));

    System.out.println(Transforms.sqrt(nd2));
    System.out.println(Transforms.exp(nd2));

    System.out.println();
    System.out.println();
    System.out.println();

    final int dimension = 10000;

    INDArray nd3 =
        Nd4j.create(
            new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    System.out.println(nd3);

//    System.out.println(nd3.broadcast(12, 3));
    System.out.println("Do something big...");

    Stopwatch sw = Stopwatch.createUnstarted();
    sw.start();
    INDArray bigND = Nd4j.rand(dimension, dimension);
    INDArray bigND2 = Nd4j.rand(dimension, dimension);
    INDArray bigND3 = bigND2.mmul(bigND);
    sw.stop();
    System.out.println("Spent " + sw.elapsed(TimeUnit.MILLISECONDS) + "ms");
    System.out.println(bigND3.rows() + " " + bigND3.columns());
//    System.out.println(bigND3.getDouble(0));
//    System.out.println(bigND3);
    System.out.println("Finish it");

    sw.reset();

    sw.start();
    bigND = Nd4j.rand(dimension, dimension);

  }

}
