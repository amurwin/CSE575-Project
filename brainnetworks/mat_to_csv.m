directory = dir('smallgraphs')

for file = 3:size(directory)
    filename = directory(file).name
    load("smallgraphs/" + "M87102217_fiber.mat")
    file = fopen("CSVdata\" + filename(1:end-4) + ".csv", "w");
    A = full(fibergraph);
    for row = 1:70
        for col = 1:69
            fprintf(file, "%.0f,",A(row, col));
        end
        fprintf(file, "%.0f\n",A(row, 70));
    end
    fclose(file);
end



