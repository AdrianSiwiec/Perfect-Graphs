/*
 * (C) Copyright 2003-2020, by Barak Naveh and Contributors.
 *
 * JGraphT : a free Java graph-theory library
 *
 * See the CONTRIBUTORS.md file distributed with this work for additional
 * information regarding copyright ownership.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the
 * GNU Lesser General Public License v2.1 or later
 * which is available at
 * http://www.gnu.org/licenses/old-licenses/lgpl-2.1-standalone.html.
 *
 * SPDX-License-Identifier: EPL-2.0 OR LGPL-2.1-or-later
 */
package hello;


import org.jgrapht.*;
import org.jgrapht.alg.cycle.BergeGraphInspector;
import org.jgrapht.graph.*;
//import org.jgrapht.nio.*;
//import org.jgrapht.nio.dot.*;
import org.jgrapht.traverse.*;

import java.io.*;
import java.net.*;
import java.rmi.server.ExportException;
import java.util.*;


/**
 * A simple introduction to using JGraphT.
 *
 * @author Barak Naveh
 */
public final class HelloJGraphT {
    private HelloJGraphT() {
    } // ensure non-instantiability.

    /**
     * The starting point for the demo.
     *
     * @param args ignored.
     * @throws URISyntaxException if invalid URI is constructed.
     * @throws ExportException    if graph cannot be exported.
     */
    public static void main(String[] args)
            throws URISyntaxException,
            IOException {
        BufferedReader reader;
        ArrayList<Graph> graphs = new ArrayList<>();
        String filePath = "/home/pierre/Documents/Studia/Magisterium/code/test/cuPerfTest/tests/java.t.in";
        reader = new BufferedReader(new FileReader(filePath));
        String line = reader.readLine();

        System.out.println("Hello!");

        while (line != null) {
            if (line.trim().length() == 0) {
                line = reader.readLine();
                continue;
            }
            int n = Integer.parseInt(line);
            String s = "";
            for (int i = 0; i < n; i++) {
                line = reader.readLine();
                s = s + line;
            }
            graphs.add(getGraphFromString(n, s));
            line = reader.readLine();
        }

        Collections.shuffle(graphs);

        long start = System.currentTimeMillis();
        for (int i = 0; i < graphs.size(); i++) {
            System.out.println("Testing on n=" + graphs.get(i).vertexSet().size());

            test(graphs.get(i));

            long now = System.currentTimeMillis();
            double elapsed = (now - start) / 1000.0;
            double sLeft = (elapsed / (i + 1)) * (graphs.size() - i - 1);
            int mLeft = (int) (sLeft / 60);
            sLeft = sLeft - mLeft * 60;
            System.out.println("Done " + (i + 1) + " of " + graphs.size());
            System.out.print("Time left: ");
            if(mLeft > 0) {
                System.out.print(mLeft + "m");
            }
            System.out.println(sLeft + "s");
        }

        System.out.println("algorithm, n, result, num_runs, Overall");
        for (int n = minN; n <= maxN; n++) {
            double avg = overallTimes.get(n) / timesCompleted.get(n) / 1000.0;
            System.out.printf("JAVA,%d,1,%d,%f%n", n, timesCompleted.get(n), avg);
        }
//        System.out.println(inspector.isBerge(g));
    }

    static BergeGraphInspector inspector = new BergeGraphInspector();
    static Map<Integer, Double> overallTimes = new HashMap<Integer, Double>();
    static Map<Integer, Integer> timesCompleted = new HashMap<>();
    static int minN = 100;
    static int maxN = 0;

    private static void test(Graph<Integer, DefaultEdge> g) {
        long start = System.currentTimeMillis();
        boolean res = inspector.isBerge(g);
        long end = System.currentTimeMillis();

        int n = g.vertexSet().size();
        minN = Math.min(minN, n);
        maxN = Math.max(maxN, n);
        overallTimes.put(n, overallTimes.getOrDefault(n, 0.0) + end - start);
        timesCompleted.put(n, timesCompleted.getOrDefault(n, 0) + 1);

        System.out.println("Took " + (end-start)/1000.0);

        if (res != true) {
            throw new RuntimeException("Not Berge!");
        }
    }

    private static Graph<Integer, DefaultEdge> getGraphFromString(int n, String s) {
        Graph<Integer, DefaultEdge> g = new SimpleGraph<>(DefaultEdge.class);
        for (int i = 0; i < n; i++) {
            g.addVertex(i);
        }
        s = s.replaceAll("\\s+", "");
        if (s.length() != n * n) {
            throw new RuntimeException("Graph init from string failed" + n + "\n" + s);
        }

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i * n + j) == 'X' || s.charAt(i * n + j) == '1') {
                    g.addEdge(i, j);
                }
            }
        }
        return g;
    }
}
