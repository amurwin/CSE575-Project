directory = dir('smallgraphs')

for file = 3:size(directory)
    filename = directory(file).name
    load("smallgraphs/" + filename)
    file = fopen("CSVdata\" + filename(1:end-4) + ".csv", "w");
    A = full(fibergraph);
    for row = 1:70
        for col = 1:70
            fprintf(file, "%.0f,",A(row, col));
        end
        fprintf(file, "\n");
    end
    fclose(file);
end



