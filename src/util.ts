export const doNTimes: <t>(n: number, callback: (index: number) => t) => t[] = <t>(n: number, treatmentCallback: (index: number) => t): t[] => {
    const result = [];
    for (let i = 0; i < n; i++) {
        result.push(treatmentCallback(i));
    }
    return result;
}


export const decimal = (n: number, limit: number) => {
    // tslint:disable-next-line: no-construct
    return (new Number(n)).toFixed(limit);
}
