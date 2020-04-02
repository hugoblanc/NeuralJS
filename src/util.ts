export const doNTimes: <t>(n: number, callback: (index: number) => t) => t[] = <t>(n: number, treatmentCallback: (index: number) => t): t[] => {
    const result = [];
    for (let i = 0; i < n; i++) {
        result.push(treatmentCallback(i));
    }
    return result;
}

