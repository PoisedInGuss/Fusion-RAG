package correlation;

import java.util.*;

public class CrossProduct {

    // Function to compute the cross product of the map
    public static List<List<Integer>> computeCrossProduct(Map<Double, List<Integer>> map) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> currentTuple = new ArrayList<>();
        computeCrossProductHelper(map, new ArrayList<>(map.keySet()), currentTuple, result);
        return result;
    }

    // Helper function to compute cross product recursively
    private static void computeCrossProductHelper(Map<Double, List<Integer>> map,
                                                  List<Double> keys,
                                                  List<Integer> currentTuple,
                                                  List<List<Integer>> result) {
        // If all keys have been processed, add the current tuple to the result
        if (keys.isEmpty()) {
            result.add(new ArrayList<>(currentTuple));
            return;
        }

        // Get the first key and its corresponding list of values
        Double currentKey = keys.get(0);
        List<Integer> currentList = map.get(currentKey);

        // Recursively build the tuples by combining each element in the current list
        for (Integer value : currentList) {
            currentTuple.add(value);  // Add the value to the current tuple
            computeCrossProductHelper(map, keys.subList(1, keys.size()), currentTuple, result);  // Recurse
            currentTuple.remove(currentTuple.size() - 1);  // Remove the last element to backtrack
        }
    }

    public static void main(String[] args) {
        // Example Map<Integer, List<Integer>>
        Map<Double, List<Integer>> map = new HashMap<>();
        map.put(1.0, Arrays.asList(1, 2));
        map.put(2.0, Arrays.asList(3, 4));
        map.put(3.0, Arrays.asList(5, 6));

        // Compute the cross product
        List<List<Integer>> crossProduct = computeCrossProduct(map);

        // Print the result
        for (List<Integer> tuple : crossProduct) {
            System.out.println(tuple);
        }
    }
}
